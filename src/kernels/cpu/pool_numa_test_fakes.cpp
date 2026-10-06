// src/kernels/cpu/pool_numa_test_fakes.cpp - synthetic expert kernels for pool_numa_test.cpp.
//
// They implement the exact symbols the real pool.cpp references, with deterministic arithmetic so a test can
// verify "every expert's rows were written exactly once, by the right node's task table" without AVX-512
// hardware or the ggml submodule.
//
// Convention: a value derived from the blob pointer identifies the expert, so a row written for the wrong
// expert (or twice) is visible as a wrong value.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cmath>
#include <cstdint>

namespace strata::kernels::cpu {

float fake_base(const void* p) {
    // 1.0 .. 4096.0, stable for a given address; collisions between the test's few blobs are astronomically
    // unlikely because the blobs are distinct multi-megabyte allocations.
    return (float) ((((uintptr_t) p >> 12) & 0xFFFu) + 1u);
}

bool cpu_avx2_ok() { return true; }
bool expert_oracle_q8_0_enabled() { return false; }

void act_quant_q8_1(const float* x, int n, ActQ& a) {
    (void) x; (void) n;
    a.nchunks = 1;
}
void act_quant_any(const float* x, int n, ActQ& a) {
    (void) x; (void) n;
    a.nchunks = 1;
}
void native_quant_act(const NativeFmt& f, const float* x, void* dst) { (void) f; (void) x; (void) dst; }
void native_quant_h(const NativeFmt& f, const float* h, void* dst) { (void) f; (void) h; (void) dst; }

void s2_expert_vnni_q(const uint8_t* blob, const ActQ&, float* out, ExpertScratch&) {
    const float b = fake_base(blob);
    for (int i = 0; i < H; ++i) out[i] = b + 2000000.0f + (float) i;
}

void s2_expert_gu_rows(const uint8_t* blob, const ActQ&, float* ff, int r0, int r1) {
    const float b = fake_base(blob);
    for (int r = r0; r < r1; ++r) ff[r] = b + (float) r;
}

void s2_expert_down_rows(const uint8_t* blob, const ActQ&, float* out, int r0, int r1) {
    const float b = fake_base(blob);
    for (int r = r0; r < r1; ++r) out[r] = b + 1000000.0f + (float) r;
}

void s2_expert_gu_rows_multi(const uint8_t* blob, const ActQ* const*, int nt, float* const* ff, int r0, int r1) {
    const float b = fake_base(blob);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) ff[t][r] = b + (float) (t * 10000) + (float) r;
}

void s2_expert_down_rows_multi(const uint8_t* blob, const ActQ* const*, int nt, float* const* out, int r0, int r1) {
    const float b = fake_base(blob);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) out[t][r] = b + 1000000.0f + (float) (t * 10000) + (float) r;
}

void native_gu_rows(const NativeFmt&, const uint8_t* blob, const void* const*, int nt, float* const* ff, int r0,
                    int r1) {
    const float b = fake_base(blob);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) ff[t][r] = b + (float) (t * 10000) + (float) r;
}

void native_down_rows(const NativeFmt&, const uint8_t* blob, const void* const*, int nt, float* const* out, int r0,
                      int r1) {
    const float b = fake_base(blob);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) out[t][r] = b + 1000000.0f + (float) (t * 10000) + (float) r;
}

void q2_rows_any(const uint8_t* w, size_t, int, const ActQ* const*, int nt, float* const* out, int r0, int r1) {
    const float b = fake_base(w);
    for (int t = 0; t < nt; ++t)
        for (int r = r0; r < r1; ++r) out[t][r] = b + 1000000.0f + (float) (t * 10000) + (float) r;
}

}  // namespace strata::kernels::cpu

// cpu_features(): the harness pretends the AVX-512 set is present so upstream tests that gate on it run.
namespace strata::kernels::cpu {
CpuFeatures cpu_features() {
    CpuFeatures f;
    f.avx512f = f.avx512bw = f.avx512vl = f.avx512_vnni = f.avx512_vbmi = true;
    return f;
}
}  // namespace strata::kernels::cpu

namespace strata::kernels::cpu {
const char* CpuFeatures::reason() const { return "ok"; }
}  // namespace strata::kernels::cpu
