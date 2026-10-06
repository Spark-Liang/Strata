// include/strata/core/expert_placement.hpp - L1 NUMA: which node's arena holds each routed expert.
//
// The pool claims per node and the arena is split per node, so the only missing fact is the placement itself:
// for every (layer, expert) which of the two arenas owns it.  The rule is deliberately boring and static:
//
//   * PER-LAYER BALANCE.  Every layer's 512 experts split 256/256, because a layer is a barrier and both
//     nodes must have work while it runs.  Assigning whole layers to a node would leave one node idle on
//     every layer and use only one socket's bandwidth at a time.
//   * OPTIONAL HOT-SET BIAS.  The GPU is on node 0; an expert the profile ranks hot is more likely to be
//     filled into the VRAM cache over PCIe, so hot experts are swapped onto node 0 where that read is local.
//   * DETERMINISTIC.  No randomness and no dependence on load order, so two starts place the same pack
//     identically (the tests compare builds byte for byte).
//
// Pure: no allocation is performed by the caller's side effects beyond the returned bitmap, and the function
// is unit-tested without a GPU (tests/core/expert_placement_test.cpp).
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

struct ExpertPlacement {
    int n_nodes = 1;
    int64_t n_layers = 0, n_expert = 0;
    std::vector<uint8_t> node;   ///< [layer * n_expert + expert] -> node index

    bool empty() const { return node.empty(); }
    int node_of(int64_t layer, int64_t expert) const {
        if (node.empty() || layer < 0 || layer >= n_layers || expert < 0 || expert >= n_expert) return 0;
        return (int) node[(size_t) (layer * n_expert + expert)];
    }
    /// Bytes the placement puts on node `nd`, from each layer's blob size.
    uint64_t bytes_of(int nd, const std::vector<uint64_t>& layer_blob_bytes) const;
};

/// Builds the placement for `n_nodes` arenas.  `layer_blob_bytes` is the layout's per-layer blob size (a
/// canonical pack repeats one value; a native pack varies).  `hot` (may be null) is an expert profile rank,
/// hottest first, as (layer, expert) pairs.  False with a reason when the geometry is inconsistent.
bool build_expert_placement(int64_t n_layers, int64_t n_expert,
                            const std::vector<uint64_t>& layer_blob_bytes,
                            const std::vector<std::pair<int32_t, int32_t>>* hot, int n_nodes,
                            ExpertPlacement& out, std::string& err);

}  // namespace strata::core
