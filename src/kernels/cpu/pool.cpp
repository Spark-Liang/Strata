// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "pool_affinity_win.hpp"
#else
#include <pthread.h>
#include <sched.h>
#include "pool_affinity_linux.hpp"
#endif

namespace strata::kernels::cpu {

namespace {
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}

#if defined(__linux__)
// L1 NUMA: the node of every CPU, read once from /sys/devices/system/node/node<N>/cpulist ("0-17,36-53", the
// format the kernel documents).  Empty on any failure, which the constructor turns into the legacy single-node
// path.  Deliberately not `physical_package_id`: it happens to equal the node on one-node-per-socket systems
// and does not on NPS2/NPS4 parts.
std::vector<int> sysfs_cpu_nodes() {
    std::vector<int> map;
    for (int node = 0; node < 1024; ++node) {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/node/node%d/cpulist", node);
        std::FILE* f = std::fopen(path, "r");
        if (f == nullptr) break;   // node ids are contiguous from 0
        char buf[8192];
        const size_t got = std::fread(buf, 1, sizeof buf - 1, f);
        std::fclose(f);
        buf[got] = '\0';
        char* p = buf;
        while (*p != '\0') {
            char* end = nullptr;
            const long lo = std::strtol(p, &end, 10);
            if (end == p) break;
            long hi = lo;
            p = end;
            if (*p == '-') {
                hi = std::strtol(p + 1, &end, 10);
                p = end;
            }
            if (lo < 0 || hi > (1 << 20) || hi < lo) break;
            if ((long) map.size() <= hi) map.resize((size_t) hi + 1, -1);
            for (long c = lo; c <= hi; ++c) map[(size_t) c] = node;
            if (*p == ',') ++p;
        }
    }
    return map;
}
#endif
}  // namespace

/// L1 NUMA: whether a batch has any job outside `host_node`.  A batch on the host node only is the
/// unpartitioned source's answer (every `ExpertJobMulti.node` defaulted to 0) or a lopsided route; both take
/// the legacy single-head path, where every worker claims - so enabling the pool on a two-node machine before
/// the arena is partitioned can only be neutral, never a regression that idles one node.
static bool batch_uses_nodes(const ExpertJobMulti* jobs, int n, int host_node) {
    for (int e = 0; e < n; ++e)
        if (jobs[e].node != host_node) return true;
    return false;
}

CpuTopology detect_cpu_topology(bool skip_first, PoolAffinity affinity) {
    CpuTopology topo;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        // Preserve group identity even when detailed core topology is unavailable.
        const WORD groups = GetActiveProcessorGroupCount();
        for (WORD group = 0; group < groups; ++group) {
            const DWORD count = GetActiveProcessorCount(group);
            if (count == 0 || count == (DWORD) -1 || count > 64) continue;
            for (DWORD i = 0; i < count; ++i) topo.worker_cores.push_back((int) group * 64 + (int) i);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }
    std::vector<char> buf(len);
    if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                         (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        const char* p = buf.data();
        const char* end = p + len;
        struct CoreDesc {
            uint8_t efficiency = 0;
            bool has_smt = false;
            std::vector<int> lps;
        };
        std::vector<CoreDesc> descs;
        while (p < end) {
            const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
            if (e->Relationship == RelationProcessorCore) {
                CoreDesc cd;
                cd.efficiency = e->Processor.EfficiencyClass;
                cd.has_smt = (e->Processor.Flags & LTP_PC_SMT) != 0;
                for (WORD group = 0; group < e->Processor.GroupCount; ++group) {
                    const GROUP_AFFINITY& g = e->Processor.GroupMask[group];
                    for (int bit = 0; bit < 64; ++bit) {
                        if (g.Mask & (KAFFINITY(1) << bit)) {
                            cd.lps.push_back((int) (g.Group * 64 + bit));
                        }
                    }
                }
                if (!cd.lps.empty()) {
                    descs.push_back(std::move(cd));
                }
            }
            p += e->Size;
        }

        uint8_t min_eff = 255, max_eff = 0;
        for (const auto& c : descs) {
            min_eff = (std::min)(min_eff, c.efficiency);
            max_eff = (std::max)(max_eff, c.efficiency);
        }

        topo.is_hybrid = (max_eff > min_eff);
        if (topo.is_hybrid) {
            for (const auto& c : descs) {
                if (c.efficiency == max_eff) {
                    topo.p_cores++;
                    topo.p_threads += (int) c.lps.size();
                } else {
                    topo.e_cores++;
                }
            }
        } else {
            topo.p_cores = (int) descs.size();
            for (const auto& c : descs) topo.p_threads += (int) c.lps.size();
        }

        if (affinity == PoolAffinity::All || !topo.is_hybrid) {
            // #642 (from Hardin22's fork): on a hybrid CPU the P-cores first (the host takes the first of them), so a
            // pool smaller than the core count (setup's --pool-workers for a hybrid CPU) runs on the P-cores and the
            // first E-cores rather than on whatever the OS numbered first.  All cores alike: the order is unchanged.
            if (topo.is_hybrid)
                std::stable_sort(descs.begin(), descs.end(),
                                 [](const CoreDesc& x, const CoreDesc& y) { return x.efficiency > y.efficiency; });
            for (const auto& c : descs) topo.worker_cores.push_back(c.lps[0]);
            if (skip_first && !topo.worker_cores.empty()) {
                topo.host_core = topo.worker_cores.front();
                topo.worker_cores.erase(topo.worker_cores.begin());
            }
            return topo;
        }

        // Hybrid CPU with Auto or PCores affinity:
        // Prioritize Performance cores:
        // 1. Primary logical processor of each P-core (avoids SMT resource contention)
        // 2. SMT sibling logical processors of P-cores
        // 3. E-cores (only as overflow in Auto mode)
        std::vector<int> p_primaries;
        std::vector<int> p_siblings;
        std::vector<int> e_cores;

        for (const auto& c : descs) {
            if (c.efficiency == max_eff) {
                p_primaries.push_back(c.lps[0]);
                for (size_t s = 1; s < c.lps.size(); ++s) {
                    p_siblings.push_back(c.lps[s]);
                }
            } else {
                for (int lp : c.lps) e_cores.push_back(lp);
            }
        }

        if (skip_first && !p_primaries.empty()) {
            topo.host_core = p_primaries.front();
            p_primaries.erase(p_primaries.begin());
        }

        for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
        for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
        if (affinity != PoolAffinity::PCores) {
            for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
        }
        return topo;
    }
#else
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE (issue #40): SMT siblings share a core's
    // load/store bandwidth, so a worker on each would put two workers on one core, as the Windows branch above
    // explains.  sysfs names each CPU's (package, core); the first allowed CPU of each pair is kept, so a taskset
    // that leaves out the first sibling still gets its core.  Without sysfs every allowed CPU counts, as before.
    auto topo_read = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    auto cap_read = [](int cpu) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };

    std::vector<int> allowed;
    std::vector<unsigned long> allowed_mask;
    if (detail::get_thread_affinity(allowed_mask, &allowed) != 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }

    struct CoreLinux {
        int cpu = -1;
        long pkg = -1;
        long core = -1;
        long cap = -1;
        bool is_sibling = false;
    };
    std::vector<CoreLinux> all_cpus;
    std::vector<std::pair<long, long>> seen_phys;
    long max_cap = 0, min_cap = 1000000;

    for (int cpu : allowed) {
        CoreLinux cl;
        cl.cpu = cpu;
        cl.pkg = topo_read(cpu, "physical_package_id");
        cl.core = topo_read(cpu, "core_id");
        cl.cap = cap_read(cpu);
        if (cl.cap > 0) {
            max_cap = (std::max)(max_cap, cl.cap);
            min_cap = (std::min)(min_cap, cl.cap);
        }
        if (cl.pkg >= 0 && cl.core >= 0) {
            const std::pair<long, long> key{cl.pkg, cl.core};
            if (std::find(seen_phys.begin(), seen_phys.end(), key) != seen_phys.end()) {
                cl.is_sibling = true;
            } else {
                seen_phys.push_back(key);
            }
        }
        all_cpus.push_back(cl);
    }

    topo.is_hybrid = (max_cap > 0 && max_cap > min_cap);
    if (topo.is_hybrid) {
        for (const auto& cl : all_cpus) {
            if (cl.cap == max_cap) {
                if (!cl.is_sibling) topo.p_cores++;
                topo.p_threads++;
            } else {
                if (!cl.is_sibling) topo.e_cores++;
            }
        }
    } else {
        topo.p_cores = (int) seen_phys.size();
        topo.p_threads = (int) all_cpus.size();
    }

    if (affinity == PoolAffinity::All || !topo.is_hybrid) {
        if (topo.is_hybrid)   // #642: the P-cores first (see the Windows branch)
            std::stable_sort(all_cpus.begin(), all_cpus.end(),
                             [](const CoreLinux& x, const CoreLinux& y) { return x.cap > y.cap; });
        for (const auto& cl : all_cpus) {
            if (!cl.is_sibling) topo.worker_cores.push_back(cl.cpu);
        }
        if (skip_first && !topo.worker_cores.empty()) {
            topo.host_core = topo.worker_cores.front();
            topo.worker_cores.erase(topo.worker_cores.begin());
        }
        return topo;
    }

    // Hybrid CPU on Linux:
    std::vector<int> p_primaries;
    std::vector<int> p_siblings;
    std::vector<int> e_cores;

    for (const auto& cl : all_cpus) {
        if (cl.cap == max_cap) {
            if (!cl.is_sibling) p_primaries.push_back(cl.cpu);
            else p_siblings.push_back(cl.cpu);
        } else {
            e_cores.push_back(cl.cpu);
        }
    }

    if (skip_first && !p_primaries.empty()) {
        topo.host_core = p_primaries.front();
        p_primaries.erase(p_primaries.begin());
    }

    for (int cpu : p_primaries) topo.worker_cores.push_back(cpu);
    for (int cpu : p_siblings) topo.worker_cores.push_back(cpu);
    if (affinity != PoolAffinity::PCores) {
        for (int cpu : e_cores) topo.worker_cores.push_back(cpu);
    }
    return topo;
#endif
    return topo;
}

std::vector<int> physical_cores(bool skip_first, PoolAffinity affinity) {
    return detect_cpu_topology(skip_first, affinity).worker_cores;
}

namespace {

bool pin_this_thread(int core, [[maybe_unused]] int worker = -1) {
    if (core < 0) return false;
#if defined(_WIN32)
    return detail::set_thread_group_affinity(core, worker);
#else
    const int error = detail::pin_thread_to_cpu(core);
    if (error != 0)
        std::fprintf(stderr, "strata cpu pool: affinity for worker %d (CPU %d) failed: %d; previous affinity kept\n",
                     worker, core, error);
    return error == 0;
#endif
}

}  // namespace

ThreadAffinity pin_current_thread(int core) {
    if (core < 0) return {};
#if defined(_WIN32)
    ThreadAffinity previous;
    ULONG target = 0;
    if (!detail::get_thread_cpu_sets(previous.cpu_sets) || !detail::cpu_set_for_core(core, target) ||
        !SetThreadSelectedCpuSets(GetCurrentThread(), &target, 1)) {
        std::fprintf(stderr, "strata cpu pool: host CPU Set selection for processor %d failed: %lu; previous placement kept\n",
                     core, (unsigned long) GetLastError());
        return {};
    }
    previous.valid = true;
    return previous;
#else
    ThreadAffinity previous;
    int error = detail::get_thread_affinity(previous.mask);
    if (error == 0) error = detail::pin_thread_to_cpu(core);
    if (error != 0) {
        std::fprintf(stderr, "strata cpu pool: host affinity for CPU %d failed: %d; previous affinity kept\n", core, error);
        return {};
    }
    previous.valid = true;
    return previous;
#endif
}

void restore_thread_affinity(const ThreadAffinity& previous) {
    if (!previous.valid) return;
#if defined(_WIN32)
    // Clearing an originally empty selection restores process-default/all-group eligibility without
    // turning the caller's implicit Windows 11 affinity into an explicit single-group hard mask.
    if (!SetThreadSelectedCpuSets(GetCurrentThread(), previous.cpu_sets.empty() ? nullptr : previous.cpu_sets.data(),
                                 (ULONG) previous.cpu_sets.size()))
        std::fprintf(stderr, "strata cpu pool: host CPU Set restoration failed: %lu\n",
                     (unsigned long) GetLastError());
#else
    const int error = detail::set_thread_affinity(previous.mask);
    if (error != 0)
        std::fprintf(stderr, "strata cpu pool: host affinity restoration failed: %d\n", error);
#endif
}

namespace {
std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    const uint64_t h = head_[0].load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d, nodes %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_, n_nodes_);
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works, PoolAffinity affinity)
    : host_works_(host_works), affinity_(affinity), topo_(detect_cpu_topology(true, affinity)) {
    if (const char* e = std::getenv("STRATA_POOL_SPIN_US"))   // a test knob; see kSpinBeforeSleep
        spin_before_sleep_ = std::chrono::microseconds((std::max)(0, std::atoi(e)));
    if (const char* e = std::getenv("STRATA_POOL_TASKS_PER_THREAD"))   // phase-granularity knob: 3 = upstream
        tasks_per_thread_ = (std::min)(64, (std::max)(1, std::atoi(e)));
    if (n_workers > 0) {
        n_ = n_workers;
    } else if (topo_.is_hybrid && affinity_ != PoolAffinity::All) {
        n_ = (std::max)(1, topo_.p_cores - 1);
    } else {
        n_ = (int) topo_.worker_cores.size();
    }
    if (n_ < 1) n_ = 1;
    // ---- L1 NUMA: which node each worker sits on.  STRATA_POOL_LOGICAL_NODES=2 is a test knob that splits the
    // workers round-robin over two logical nodes on any machine, so the two-head claim protocol is what it
    // stresses; otherwise the node comes from sysfs and any failure keeps the legacy single-head path.
    if (const char* v = std::getenv("STRATA_POOL_LOGICAL_NODES")) {
        const int k = std::atoi(v);
        if (k >= 2 && k <= kMaxNodes && k <= n_) {
            n_nodes_ = k;
            worker_node_.resize((size_t) n_, 0);
            for (int i = 0; i < n_; ++i) worker_node_[(size_t) i] = i % k;
        }
    }
#if defined(__linux__)
    if (n_nodes_ == 1) {
        const std::vector<int> cpu_node = sysfs_cpu_nodes();
        if (!cpu_node.empty()) {
            std::vector<int> wn((size_t) n_, 0);
            int max_node = 0;
            bool ok = true;
            for (int i = 0; i < n_; ++i) {
                const int c = i < (int) topo_.worker_cores.size() ? topo_.worker_cores[(size_t) i] : -1;
                if (c < 0 || c >= (int) cpu_node.size() || cpu_node[(size_t) c] < 0) { ok = false; break; }
                wn[(size_t) i] = cpu_node[(size_t) c];
                max_node = (std::max)(max_node, wn[(size_t) i]);
            }
            // More than one node is used only when every node with worker tasks has a claiming thread: a job on
            // a node nobody claims would never complete.  A node whose only claimant would be the host counts
            // only when the host works.
            if (ok && max_node + 1 >= 2 && max_node + 1 <= kMaxNodes) {
                int host_cpu_node = -1;
                if (topo_.host_core >= 0 && topo_.host_core < (int) cpu_node.size())
                    host_cpu_node = cpu_node[(size_t) topo_.host_core];
                bool each = true;
                for (int nd = 0; nd < max_node + 1; ++nd) {
                    bool has = false;
                    for (int i = 0; i < n_; ++i) has = has || wn[(size_t) i] == nd;
                    if (host_works_ && host_cpu_node == nd) has = true;
                    each = each && has;
                }
                if (each) {
                    n_nodes_ = max_node + 1;
                    worker_node_ = std::move(wn);
                }
                if (host_cpu_node >= 0) host_node_ = host_cpu_node;
            }
        }
    }
#endif
    worker_node_.resize((size_t) n_, 0);   // the legacy path: every worker on node 0, `n_nodes_` stays 1
    scratch_.resize((size_t) n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    strata::core::diag_pool_fn().store(&diag_active_pool);
    split_.resize((size_t) kMaxSplit);
    split_multi_.resize((size_t) kMaxSplitMulti);
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) topo_.worker_cores.size() ? topo_.worker_cores[(size_t) i] : -1) : -1;
        threads_.emplace_back([this, i, core] {
            pin_this_thread(core, i);
            worker(i);
        });
    }
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
    for (auto& t : threads_) t.join();
}

int ExpertPool::node_workers(int node) const {
    if (node < 0 || node >= kMaxNodes) return 0;
    int c = 0;
    for (int w : worker_node_) c += (w == node);
    if (host_works_ && host_node_ == node) ++c;
    return c;
}

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction.  Counting only
    // on the RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first
    // `run()` - which waits for `parked_ == n_` before publishing - then deadlocks.  It deadlocks on the very
    // first call, which is the good case; a version that deadlocked on the second would be far worse.
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        // Park: wait for work.  `_mm_pause` rather than a bare spin because it yields the pipeline to the
        // sibling hyperthread; `epoch_` is bumped once per LAYER, not once per expert, so most of these
        // iterations are spent here with nothing to do.
        //
        // **AND NOTHING ELSE HAPPENS IN HERE.**  This loop used to do `pauses_.fetch_add(1)` on every iteration
        // - a locked read-modify-write, five workers against one cache line - so the workers spent their wait
        // invalidating each other's caches and the very line the host writes to publish work.  The counter was
        // diagnostic and nothing branched on it.  See the note on the atomics in pool.hpp.
        //
        // After `kSpinBeforeSleep` with no work the worker sleeps instead (issue #4).  The clock is read once
        // every 1024 pauses, so the spin itself is unchanged.
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            if (std::chrono::steady_clock::now() - parked_at < spin_before_sleep_) continue;
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.  Only
        // this epoch's jobs: if the host has already moved on, the claims fail and the worker parks again.
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        // L1 NUMA: a per-node batch claims only this worker's node's tasks; a legacy batch keeps the single-head
        // order, which is what the fields `jobs_`/`mjobs_` still describe.
        const int drain_node = per_node_batch_ ? worker_node_[(size_t) id] : 0;
        drain(id, scratch_[(size_t) id], seen, drain_node);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

int ExpertPool::claim(uint32_t epoch, int node, int64_t& row0, int64_t& row1) {
    if (node < 0 || node >= kMaxNodes) node = 0;
    uint64_t h = head_[node].load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // this node's table is exhausted
        if (head_[node].compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            const PhaseTask& t = tasks_[node][i];
            row0 = t.row0;
            row1 = t.row1;
            return (int) i;
        }
    }
}

int ExpertPool::claim_index(uint32_t epoch) {
    uint64_t h = head_[0].load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_[0].compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        std::fprintf(stderr, "strata: expert pool batch of %d jobs is out of range\n", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    if (per_node_batch_) {
        // L1 NUMA: every node's table is described, then one publish wakes all the workers.  A node with no
        // tasks gets a head with n = 0, so its workers claim nothing and re-park - the empty-node case.
        uint32_t total = 0;
        for (int nd = 0; nd < n_nodes_; ++nd) {
            const uint32_t nt = (uint32_t) tasks_[nd].size();
            total += nt;
            head_[nd].store(pack_head(e, nt, 0), std::memory_order_release);
        }
        batch_tasks_ = total;
    } else {
        head_[0].store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
        batch_tasks_ = (uint32_t) n;
    }
    publish();
    return e;
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0;
    std::chrono::steady_clock::time_point t0{};
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u) t0 = now;
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again (issue #29)\n",
                         what, parked_.load(), n_);
            strata::core::release_gpu_waits(stderr);   // #267: the GPU may be spinning on this layer's flag
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0, seen = 0;
    std::chrono::steady_clock::time_point t0{};
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;                                // `>=`: never a wait that an overshoot outlives
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u || d != seen) { t0 = now; seen = d; }     // progress restarts the clock
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again (issue #29)\n",
                         d, n, parked_.load(), n_);
            strata::core::release_gpu_waits(stderr);   // #267
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch, uint32_t epoch, int node) {
    (void) id;
    for (;;) {
        // L1 NUMA: a per-node batch claims from this node's table, which hands back a row range in that node's
        // row space.  A legacy batch keeps the single-head index claim, from which the row range is derived
        // exactly as it was before.
        const bool per_node = per_node_batch_;
        int ci = -1;
        int64_t g0 = 0, g1 = 0;
        if (per_node) {
            ci = claim(epoch, node, g0, g1);
        } else {
            ci = claim_index(epoch);
            if (ci >= 0 && mode_ >= 3) {
                const uint32_t t = (uint32_t) ci;
                g0 = mrows_ * (int64_t) t / mtasks_;
                g1 = mrows_ * (int64_t) (t + 1) / mtasks_;
            }
        }
        if (ci < 0) break;
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(jobs_[e].blob, *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(jobs_[e].blob, split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? FF : H;
            for (int64_t r = g0; r < g1;) {
                const int e = per_node ? node_experts_[node][(size_t) (r / per)] : (int) (r / per);
                const int r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 5 && q2_native_kernels(nfmt_->gu_type)) {
                    // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt_->n_embd / 64);
                    q2_rows_any(mjobs_[e].blob, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mjobs_[e].blob + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r = r0; r < r1; ++r)
                            sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                } else if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (q2_native_kernels(nfmt_->d_type)) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            for (int64_t r = g0; r < g1;) {
                const int e = per_node ? node_experts_[node][(size_t) (r / per)] : (int) (r / per);
                const int r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mjobs_[e].blob, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mjobs_[e].blob, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    wait_parked("before a phase");
    mode_ = mode;
    per_node_batch_ = false;   // a legacy phase: one head, every worker claims
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    if (host_works_) drain(-1, host_scratch_, e, 0);
    wait_done(n_tasks);
    wait_parked("after a phase");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

void ExpertPool::build_node_tasks(int mode, ExpertJobMulti* jobs, int n, int per) {
    (void) mode;
    batch_tasks_ = 0;
    per_node_batch_ = n_nodes_ > 1;
    for (int nd = 0; nd < kMaxNodes; ++nd) {
        tasks_[nd].clear();
        node_experts_[nd].clear();
        node_rows_[nd] = 0;
    }
    if (n_nodes_ <= 1) return;
    // A job whose node has no claiming thread is drained by the host's node instead: stranding it on a node
    // nobody claims would leave `done` short of the batch and stall the pool (the guard the constructor already
    // applies to the worker layout).
    for (int e = 0; e < n; ++e) {
        int nd = jobs[e].node;
        if (nd < 0 || nd >= n_nodes_ || node_workers(nd) <= 0) nd = host_node_;
        node_experts_[nd].push_back(e);
    }
    for (int nd = 0; nd < n_nodes_; ++nd) {
        const int64_t rows = (int64_t) node_experts_[nd].size() * per;
        node_rows_[nd] = rows;
        const int threads = node_workers(nd);
        if (rows <= 0 || threads <= 0) continue;
        const int nt = (std::max)(1, tasks_per_thread_ * threads);
        for (int t = 0; t < nt; ++t) {
            const int64_t r0 = rows * t / nt, r1 = rows * (t + 1) / nt;
            if (r1 > r0) tasks_[nd].push_back({r0, r1});
        }
        batch_tasks_ += (uint32_t) tasks_[nd].size();
    }
}

void ExpertPool::run_phase_nodes(int mode) {
    wait_parked("before a phase");
    mode_ = mode;
    njobs_ = (int) batch_tasks_;
    const uint32_t e = begin_batch(0);   // mode_ >= 3 with n_nodes_ > 1: publishes every node's head
    if (host_works_) drain(-1, host_scratch_, e, host_node_);
    wait_done((int) batch_tasks_);
    wait_parked("after a phase");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = 3 * threads;
    const bool nodes_batch = n_nodes_ > 1 && batch_uses_nodes(jobs, n, host_node_);
    if (nodes_batch) {
        build_node_tasks(3, jobs, n, FF);
        run_phase_nodes(3);
    } else {
        mrows_ = (int64_t) n * FF;
        run_phase(3, mtasks_);
    }
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    if (nodes_batch) {
        build_node_tasks(4, jobs, n, H);
        run_phase_nodes(4);
    } else {
        mrows_ = (int64_t) n * H;
        run_phase(4, mtasks_);
    }
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = tasks_per_thread_ * threads;
        const bool nodes_batch = n_nodes_ > 1 && batch_uses_nodes(mjobs_, nb, host_node_);
        const auto a = std::chrono::steady_clock::now();
        if (nodes_batch) {
            build_node_tasks(5, mjobs_, nb, FF);
            run_phase_nodes(5);
        } else {
            mrows_ = (int64_t) nb * FF;
            run_phase(5, mtasks_);
        }
        const auto b = std::chrono::steady_clock::now();
        for (int e = 0; e < nb; ++e)
            for (int t = 0; t < mjobs_[e].nt; ++t)
                if (q2_native_kernels(f.d_type)) act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
        const auto c = std::chrono::steady_clock::now();
        if (nodes_batch) {
            build_node_tasks(6, mjobs_, nb, H);
            run_phase_nodes(6);
        } else {
            mrows_ = (int64_t) nb * H;
            run_phase(6, mtasks_);
        }
        const auto d = std::chrono::steady_clock::now();
        ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
        ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
        ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(jobs[i].blob, *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // Wait for every worker to be parked BEFORE touching the batch, so the publish below is the only thing
    // that can move a worker into the drain loop.
    //
    // THE THREE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is
    // slow at the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_a = std::chrono::steady_clock::now();
    wait_parked("before a batch");
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    const uint32_t e = begin_batch(n);   // done, then head (release), then the epoch: the batch is described first

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) _mm_pause();`.  The host is pinned to core 0 - the core
    // `physical_cores(true)` deliberately keeps the five workers off - so for the whole drain that core was
    // idle while five cores did six cores' worth of work.  Measured before the change: 33.7 GB/s against
    // 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const int ci = claim_index(e);
            if (ci < 0) break;
            hstate_.store(ci, std::memory_order_relaxed);
            hstate_ms_.store(now_ms(), std::memory_order_relaxed);
            const ExpertJob& j = jobs_[ci];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    wait_done(n);
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    wait_parked("after a batch");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu
