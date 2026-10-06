// src/core/expert_placement.cpp - see include/strata/core/expert_placement.hpp.
#include "strata/core/expert_placement.hpp"

namespace strata::core {

uint64_t ExpertPlacement::bytes_of(int nd, const std::vector<uint64_t>& layer_blob_bytes) const {
    if (empty() || nd < 0 || nd >= n_nodes || layer_blob_bytes.size() != (size_t) n_layers) return 0;
    uint64_t bytes = 0;
    for (int64_t l = 0; l < n_layers; ++l)
        for (int64_t e = 0; e < n_expert; ++e)
            if (node_of(l, e) == nd) bytes += layer_blob_bytes[(size_t) l];
    return bytes;
}

bool build_expert_placement(int64_t n_layers, int64_t n_expert,
                            const std::vector<uint64_t>& layer_blob_bytes,
                            const std::vector<std::pair<int32_t, int32_t>>* hot, int n_nodes,
                            ExpertPlacement& out, std::string& err) {
    if (n_layers <= 0 || n_expert <= 0) {
        err = "expert placement: non-positive geometry";
        return false;
    }
    if (layer_blob_bytes.size() != (size_t) n_layers) {
        err = "expert placement: layer blob sizes do not match the layer count";
        return false;
    }
    if (n_nodes < 1 || n_nodes > 2) {
        err = "expert placement: the L1 partition supports 1 or 2 nodes";
        return false;
    }
    out = ExpertPlacement{};
    out.n_nodes = n_nodes;
    out.n_layers = n_layers;
    out.n_expert = n_expert;
    out.node.assign((size_t) (n_layers * n_expert), 0);
    if (n_nodes == 1) return true;

    // 1) Per-layer 256/256 by expert id.  Every layer's barrier therefore has both nodes busy, and the
    //    per-node bytes differ only by the (equal) per-layer blob, so capacity is balanced within one blob.
    for (int64_t l = 0; l < n_layers; ++l)
        for (int64_t e = 0; e < n_expert; ++e)
            out.node[(size_t) (l * n_expert + e)] = (uint8_t) (e & 1);

    if (hot == nullptr || hot->empty()) return true;

    // 2) Hot-set bias: mark hot experts so they are never chosen as the cold swap partner, then move the hot
    //    ones that sit on node 1 by swapping them with a non-hot node-0 expert of the same layer.  A swapped
    //    partner is marked too, so the loop cannot oscillate.  A hot entry that is already on node 0 needs
    //    nothing; an out-of-range or duplicated entry is ignored (the profile is an optimization, not a
    //    correctness input).
    //
    //    AT MOST HALF OF A LAYER IS TREATED AS HOT, whatever list the caller passes.  The learned profile ranks
    //    EVERY pair, so a caller that hands the whole ranking over would otherwise mark both nodes' experts hot
    //    and leave no node-0 candidate to swap with - the bias silently doing nothing.  The cap keeps the
    //    layer's 256/256 split and guarantees swap candidates.
    std::vector<uint8_t> is_hot((size_t) (n_layers * n_expert), 0);
    std::vector<int32_t> layer_hot((size_t) n_layers, 0);
    const int32_t layer_hot_max = (int32_t) (n_expert / 2);
    for (const auto& he : *hot) {
        const int64_t l = he.first, e = he.second;
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) continue;
        const size_t idx = (size_t) (l * n_expert + e);
        if (is_hot[idx]) continue;
        if (layer_hot[(size_t) l] >= layer_hot_max) continue;
        is_hot[idx] = 1;
        ++layer_hot[(size_t) l];
    }
    for (const auto& he : *hot) {
        const int64_t l = he.first, e = he.second;
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) continue;
        const size_t idx = (size_t) (l * n_expert + e);
        if (out.node[idx] == 0) continue;
        int64_t cand = -1;
        for (int64_t c = 0; c < n_expert; ++c) {
            const size_t ci = (size_t) (l * n_expert + c);
            if (out.node[ci] == 0 && is_hot[ci] == 0) { cand = c; break; }
        }
        if (cand < 0) continue;                      // node 0 is all hot already: nothing to trade
        const size_t ci = (size_t) (l * n_expert + cand);
        out.node[idx] = 0;
        out.node[ci] = 1;
        is_hot[ci] = 1;                              // the partner must not be swapped again
    }
    return true;
}

}  // namespace strata::core
