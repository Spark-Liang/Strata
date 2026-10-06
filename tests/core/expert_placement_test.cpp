// tests/core/expert_placement_test.cpp - L1 NUMA placement: balance, determinism, hot bias.
#include "strata/core/expert_placement.hpp"

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace core = strata::core;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

}  // namespace

int main() {
    // ---- canonical pack: 48 layers x 512 experts, one blob size
    {
        const int64_t L = 48, E = 512;
        std::vector<uint64_t> blob((size_t) L, 1382400ull);
        core::ExpertPlacement p;
        std::string err;
        check(core::build_expert_placement(L, E, blob, nullptr, 2, p, err), "canonical builds");
        bool balanced = true;
        for (int64_t l = 0; l < L; ++l) {
            int c0 = 0;
            for (int64_t e = 0; e < E; ++e) c0 += p.node_of(l, e) == 0;
            balanced = balanced && c0 == E / 2;
        }
        check(balanced, "every layer splits 256/256");
        check(p.bytes_of(0, blob) == p.bytes_of(1, blob), "canonical bytes equal per node");
        // determinism
        core::ExpertPlacement q;
        check(core::build_expert_placement(L, E, blob, nullptr, 2, q, err), "canonical rebuilds");
        check(p.node == q.node, "deterministic");
    }
    // ---- native pack: varying per-layer blob sizes; per-layer counts still balance
    {
        const int64_t L = 48, E = 512;
        std::vector<uint64_t> blob((size_t) L);
        for (int64_t l = 0; l < L; ++l) blob[(size_t) l] = 1300000ull + (uint64_t) (l % 7) * 90000;
        core::ExpertPlacement p;
        std::string err;
        check(core::build_expert_placement(L, E, blob, nullptr, 2, p, err), "native builds");
        bool balanced = true;
        for (int64_t l = 0; l < L; ++l) {
            int c0 = 0;
            for (int64_t e = 0; e < E; ++e) c0 += p.node_of(l, e) == 0;
            balanced = balanced && c0 == E / 2;
        }
        check(balanced, "native layers split 256/256");
        const uint64_t b0 = p.bytes_of(0, blob), b1 = p.bytes_of(1, blob);
        const uint64_t diff = b0 > b1 ? b0 - b1 : b1 - b0;
        uint64_t max_blob = 0;
        for (uint64_t b : blob) max_blob = b > max_blob ? b : max_blob;
        check(diff <= max_blob / 2, "native node bytes balance within half a layer blob");
    }
    // ---- hot bias: every hot expert that can be traded ends on node 0
    {
        const int64_t L = 48, E = 512;
        std::vector<uint64_t> blob((size_t) L, 1382400ull);
        std::vector<std::pair<int32_t, int32_t>> hot;
        for (int64_t l = 0; l < L; ++l)
            for (int64_t e = 1; e < 20; e += 2) hot.emplace_back((int32_t) l, (int32_t) e);   // all odd ids
        core::ExpertPlacement p;
        std::string err;
        check(core::build_expert_placement(L, E, blob, &hot, 2, p, err), "hot builds");
        bool all_hot_local = true;
        for (const auto& he : hot) all_hot_local = all_hot_local && p.node_of(he.first, he.second) == 0;
        check(all_hot_local, "hot experts land on node 0");
        bool balanced = true;
        for (int64_t l = 0; l < L; ++l) {
            int c0 = 0;
            for (int64_t e = 0; e < E; ++e) c0 += p.node_of(l, e) == 0;
            balanced = balanced && c0 == E / 2;
        }
        check(balanced, "hot bias keeps the 256/256 balance");
    }
    // ---- single node: everything on node 0
    {
        std::vector<uint64_t> blob(4, 100);
        core::ExpertPlacement p;
        std::string err;
        check(core::build_expert_placement(4, 8, blob, nullptr, 1, p, err), "single node builds");
        bool all = true;
        for (int64_t l = 0; l < 4; ++l)
            for (int64_t e = 0; e < 8; ++e) all = all && p.node_of(l, e) == 0;
        check(all, "single node places everything on node 0");
    }
    // ---- full-rank hot list: the learned profile ranks every pair; the bias must still fire (top half per
    //      layer moves to node 0) instead of marking both nodes hot and doing nothing.
    {
        const int64_t L = 4, E = 512;
        std::vector<uint64_t> blob((size_t) L, 1382400ull);
        std::vector<std::pair<int32_t, int32_t>> all;
        for (int64_t l = 0; l < L; ++l)
            for (int64_t e = 0; e < E; ++e) all.emplace_back((int32_t) l, (int32_t) e);
        core::ExpertPlacement p;
        std::string err;
        check(core::build_expert_placement(L, E, blob, &all, 2, p, err), "full-rank profile builds");
        // layer 0's hot set is experts 0..255 by rank; the odd ones start on node 1 and must be swapped in
        check(p.node_of(0, 1) == 0 && p.node_of(0, 3) == 0 && p.node_of(0, 255) == 0,
              "full-rank profile still biases the hot half onto node 0");
        bool balanced = true;
        for (int64_t l = 0; l < L; ++l) {
            int c0 = 0;
            for (int64_t e = 0; e < E; ++e) c0 += p.node_of(l, e) == 0;
            balanced = balanced && c0 == E / 2;
        }
        check(balanced, "full-rank profile keeps 256/256");
    }
    // ---- bad input is refused
    {
        core::ExpertPlacement p;
        std::string err;
        check(!core::build_expert_placement(4, 8, std::vector<uint64_t>(3, 1), nullptr, 2, p, err),
              "mismatched layer sizes refused");
        check(!core::build_expert_placement(4, 8, std::vector<uint64_t>(4, 1), nullptr, 3, p, err),
              "three nodes refused");
    }
    if (failures != 0) {
        std::fprintf(stderr, "expert_placement_test: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("expert_placement_test: PASS\n");
    return 0;
}
