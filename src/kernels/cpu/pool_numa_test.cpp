// src/kernels/cpu/pool_numa_test.cpp - L1 NUMA: the expert pool's two-node claim protocol under load.
//
// The REAL pool.cpp runs here; the kernels are synthetic (pool_numa_test_fakes.cpp) and write address-derived
// values, so the test checks the parts that the NUMA change touches, on any machine and without AVX-512 or the
// ggml submodule:
//   1. every expert's rows are written exactly once, by whichever node's task table claimed them (poisoned
//      outputs catch a dropped row; a double claim leaves another row poisoned);
//   2. batches run with jobs on both nodes, all jobs on one node (the empty-node case) and with random
//      interleaving - the second node's head is an empty table there, and its workers must re-park instead of
//      stalling the barrier;
//   3. the legacy single-head paths (run/run_split, and a batch whose jobs all sit on the host's node) still
//      work while the pool is two-node;
//   4. no batch hangs: a watchdog thread turns a stall into a failure.
//
//   pool_numa_test [iterations]        (default 1500)
#include "strata/kernels/cpu/pool.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace strata::kernels::cpu {
float fake_base(const void* p);
}

namespace {

void set_test_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

std::atomic<long long> g_beat{0};
std::atomic<bool> g_stop{false};

void watchdog(int seconds) {
    auto t0 = std::chrono::steady_clock::now();
    long long last = g_beat.load();
    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        if (g_beat.load() != last) {
            last = g_beat.load();
            t0 = std::chrono::steady_clock::now();
            continue;
        }
        if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(seconds)) {
            std::fprintf(stderr, "pool_numa_test: no progress for %d s - pool stalled\n", seconds);
            std::abort();
        }
    }
}

struct Case {
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<cpu::ExpertJobMulti> jobs;
    std::vector<std::vector<float>> outs;
    std::vector<std::vector<cpu::ActQ>> acts;
    int nt = 2;

    void build(int n, int nt_in, const std::vector<int>& nodes) {
        nt = nt_in;
        blobs.assign((size_t) n, std::vector<uint8_t>(cpu::BLOB, 0));
        for (size_t e = 0; e < blobs.size(); ++e) blobs[e][0] = (uint8_t) e;   // distinct pages for distinct blobs
        jobs.assign((size_t) n, cpu::ExpertJobMulti{});
        outs.assign((size_t) n, std::vector<float>((size_t) nt * cpu::H));
        acts.assign((size_t) n, std::vector<cpu::ActQ>((size_t) nt));
        for (int e = 0; e < n; ++e) {
            jobs[(size_t) e].blob = blobs[(size_t) e].data();
            jobs[(size_t) e].node = nodes[(size_t) e];
            jobs[(size_t) e].nt = nt;
            for (int t = 0; t < nt; ++t) {
                jobs[(size_t) e].act[t] = &acts[(size_t) e][(size_t) t];
                jobs[(size_t) e].out[t] = outs[(size_t) e].data() + (size_t) t * cpu::H;
            }
        }
    }

    void poison() {
        for (auto& o : outs)
            for (float& v : o) v = std::nanf("");
    }

    bool verify() const {
        for (size_t e = 0; e < jobs.size(); ++e) {
            const float b = cpu::fake_base(jobs[e].blob);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < cpu::H; ++r) {
                    const float want = b + 1000000.0f + (float) (t * 10000) + (float) r;
                    const float got = outs[e][(size_t) t * cpu::H + (size_t) r];
                    if (!std::isfinite(got) || got != want) {
                        std::fprintf(stderr, "verify: job %zu token %d row %d: got %.3f want %.3f\n", e, t, r,
                                     (double) got, (double) want);
                        return false;
                    }
                }
        }
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    const int iters = argc > 1 ? std::atoi(argv[1]) : 1500;
    set_test_env("STRATA_POOL_LOGICAL_NODES", "2");   // split this machine's workers over two logical nodes
    set_test_env("STRATA_POOL_SPIN_US", "500");       // let workers sleep between batches, exercising late wake-ups
    // pin=false: the logical split is what this test is about; pinning stays the deployment's concern
    cpu::ExpertPool pool(4, /*pin=*/false, /*host_works=*/true);
    if (pool.nodes() != 2 || pool.node_workers(0) < 1 || pool.node_workers(1) < 1) {
        std::fprintf(stderr, "pool_numa_test: expected 2 logical nodes with workers on both, got %d (%d/%d)\n",
                     pool.nodes(), pool.node_workers(0), pool.node_workers(1));
        return 1;
    }
    std::printf("pool_numa_test: %d workers (%d node0 / %d node1, host node %d), %d iterations\n", pool.workers(),
                pool.node_workers(0), pool.node_workers(1), 0, iters);

    std::thread wd(watchdog, 30);

    Case c;
    // ---- 1. a fixed two-node batch: 3-node0 + 3-node1 experts, two tokens each
    {
        c.build(6, 2, {0, 1, 0, 1, 0, 1});
        c.poison();
        pool.run_split_multi(c.jobs.data(), (int) c.jobs.size());
        if (!c.verify()) return 1;
        g_beat.fetch_add(1);
    }
    // ---- 2. the empty-node case: every job on node 0 (the guard sends this batch down the legacy path)
    {
        c.build(4, 2, {0, 0, 0, 0});
        c.poison();
        pool.run_split_multi(c.jobs.data(), (int) c.jobs.size());
        if (!c.verify()) return 1;
        g_beat.fetch_add(1);
    }
    // ---- 3. every job on node 1: node 0's table is empty while node 1 drains
    {
        c.build(4, 1, {1, 1, 1, 1});
        c.poison();
        pool.run_split_multi(c.jobs.data(), (int) c.jobs.size());
        if (!c.verify()) return 1;
        g_beat.fetch_add(1);
    }
    // ---- 4. the legacy single-token split path still works while the pool is two-node
    {
        std::vector<std::vector<uint8_t>> blobs(5, std::vector<uint8_t>(cpu::BLOB, 0));
        std::vector<float> outs((size_t) 5 * cpu::H);
        cpu::ActQ act;
        std::vector<cpu::ExpertJob> jobs(5);
        for (int e = 0; e < 5; ++e) {
            jobs[(size_t) e].blob = blobs[(size_t) e].data();
            jobs[(size_t) e].act = &act;
            jobs[(size_t) e].out = outs.data() + (size_t) e * cpu::H;
        }
        pool.run_split(jobs.data(), (int) jobs.size());
        for (int e = 0; e < 5; ++e) {
            const float b = cpu::fake_base(blobs[(size_t) e].data());
            for (int r = 0; r < cpu::H; ++r)
                if (outs[(size_t) e * cpu::H + (size_t) r] != b + 1000000.0f + (float) r) {
                    std::fprintf(stderr, "legacy run_split: job %d row %d wrong\n", e, r);
                    return 1;
                }
        }
        g_beat.fetch_add(1);
    }
    // ---- 5. randomized stress: batch size 1..8, nt 1..2, nodes random; occasional full-node batches
    {
        std::mt19937 rng(20261005u);
        Case r;
        for (int it = 0; it < iters; ++it) {
            const int n = 1 + (int) (rng() % 8);
            const int nt = 1 + (int) (rng() % 2);
            std::vector<int> nodes((size_t) n);
            const int mode = (int) (rng() % 8);
            for (int e = 0; e < n; ++e)
                nodes[(size_t) e] = mode == 0 ? 0 : (mode == 1 ? 1 : (int) (rng() % 2));
            r.build(n, nt, nodes);
            r.poison();
            pool.run_split_multi(r.jobs.data(), n);
            if (!r.verify()) return 1;
            g_beat.fetch_add(1);
        }
    }
    g_stop = true;
    wd.join();
    std::printf("pool_numa_test: PASS (%d batches, bytes %lld)\n", iters + 4, (long long) pool.multi_bytes);
    return 0;
}
