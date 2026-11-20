// CUDA compute backend for the qwen4exp architecture (Qwen3.8-Flash-Next GGUF).
//
// Stage 1: clean, numerically exact CUDA port of qwen4_vk.cpp using pinned host memory,
// dedicated CUDA streams, native GGUF weight codecs, hybrid VRAM-cache + CPU AVX-512 MoE,
// dynamic FP16 KV caches, QSA sparse attention, PLE n-gram embeddings, and MTP speculative decoding.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cpu_kernels.h"
#include "expert_predictor.h"
#include "qwen4.h"
#include "qwen4_qsa.h"

uint64_t qw_system_free_vram(const char* device_name);  // vram_query.cpp: over all processes, 0 = unknown

#define CUDA_CHECK(x)                                                                                               \
    do {                                                                                                            \
        cudaError_t err_ = (x);                                                                                     \
        if (err_ != cudaSuccess)                                                                                    \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_) + " at " #x);           \
    } while (0)

namespace {

constexpr int NMAX = 1024;
constexpr int NT = 8;
constexpr int GEMM_MIN = 9;
constexpr int EXPERT_GEMM_MIN = 33;
constexpr int ATT_PART_MAX = 2048;
constexpr int MAX_FILLS = 64;
constexpr uint32_t FILL_MARGIN = 2;
constexpr int KV_START = 16384;
constexpr size_t VRAM_MARGIN = 2048ull << 20;
constexpr size_t CHUNK = 1ull << 30;
constexpr size_t STAGING = 256ull << 20;
constexpr float DRAFT_P_MIN = 0.5f;

enum class Codec { Q8, F32, Q51, Q4K, Q5K };

struct ExpertEntry {
    uint64_t base;
    uint32_t start, count;
};
static_assert(sizeof(ExpertEntry) == 16, "ExpertEntry layout");

struct AttnDynParams {
    uint32_t pos, chunk, n_split;
};

// =============================================================================================
// Device math & warp helpers
// =============================================================================================
__device__ __forceinline__ float d_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float d_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float d_softplus(float x) {
    if (x > 20.0f) return x;
    const float e = expf(x);
    return e < 0.01f ? e * (1.0f - e * (0.5f - e * (1.0f / 3.0f))) : logf(1.0f + e);
}

__device__ __forceinline__ float4 f4_add(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ __forceinline__ float4 f4_mad(float4 a, float s, float4 b) {
    return make_float4(fmaf(a.x, s, b.x), fmaf(a.y, s, b.y), fmaf(a.z, s, b.z), fmaf(a.w, s, b.w));
}
__device__ __forceinline__ float4 f4_scale_add(float4 a, float s, float o) {
    return make_float4(fmaf(a.x, s, o), fmaf(a.y, s, o), fmaf(a.z, s, o), fmaf(a.w, s, o));
}
__device__ __forceinline__ float f4_dot(float4 a, float4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

__device__ __forceinline__ float warp_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 8);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 2);
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 1);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

__device__ __forceinline__ uint32_t warp_min_u32(uint32_t v) {
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 8));
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 4));
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 2));
    v = min(v, __shfl_xor_sync(0xFFFFFFFFu, v, 1));
    return v;
}

__device__ __forceinline__ float2 unpack_h2(uint32_t u) {
    __half2 h;
    memcpy(&h, &u, 4);
    return __half22float2(h);
}

__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

// =============================================================================================
// GGUF weight codecs (matches shaders/q4_codec.glsl)
// =============================================================================================
__device__ __forceinline__ float4 nib4(uint32_t w, uint32_t sh) {
    return make_float4((float)((w >> sh) & 15u), (float)((w >> (8u + sh)) & 15u), (float)((w >> (16u + sh)) & 15u),
                       (float)((w >> (24u + sh)) & 15u));
}

__device__ __forceinline__ float4 hi4(uint32_t w, uint32_t b) {
    return make_float4((float)((w >> b) & 1u) * 16.0f, (float)((w >> (b + 1u)) & 1u) * 16.0f,
                       (float)((w >> (b + 2u)) & 1u) * 16.0f, (float)((w >> (b + 3u)) & 1u) * 16.0f);
}

__device__ __forceinline__ float4 i8x4(uint32_t w) {
    return make_float4((float)(int8_t)(w & 0xFFu), (float)(int8_t)((w >> 8u) & 0xFFu),
                       (float)(int8_t)((w >> 16u) & 0xFFu), (float)(int8_t)((w >> 24u) & 0xFFu));
}

__device__ __forceinline__ uint32_t kbyte(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t i) {
    const uint32_t word = (i >> 2) == 0u ? s0 : ((i >> 2) == 1u ? s1 : s2);
    return (word >> ((i & 3u) * 8u)) & 0xFFu;
}

__device__ __forceinline__ uint2 scale_min(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t j) {
    if (j < 4u) return make_uint2(kbyte(s0, s1, s2, j) & 63u, kbyte(s0, s1, s2, j + 4u) & 63u);
    const uint32_t b = kbyte(s0, s1, s2, j + 4u);
    return make_uint2((b & 15u) | ((kbyte(s0, s1, s2, j - 4u) >> 6) << 4),
                      (b >> 4) | ((kbyte(s0, s1, s2, j) >> 6) << 4));
}

template <Codec C>
__device__ __forceinline__ void decode8(uint64_t w, uint64_t s, uint32_t cols, uint32_t row, uint32_t u, float4& a0, float4& a1) {
    if constexpr (C == Codec::Q8) {
        const uint2 wv = reinterpret_cast<const uint2*>(w)[row * (cols >> 3) + u];
        const float sc = reinterpret_cast<const float*>(s)[row * (cols >> 5) + (u >> 2)];
        a0 = f4_scale_add(i8x4(wv.x), sc, 0.0f);
        a1 = f4_scale_add(i8x4(wv.y), sc, 0.0f);
    } else if constexpr (C == Codec::F32) {
        const float4* W = reinterpret_cast<const float4*>(w);
        const uint32_t b = row * (cols >> 2) + u * 2u;
        a0 = W[b];
        a1 = W[b + 1u];
    } else if constexpr (C == Codec::Q51) {
        const uint32_t* W = reinterpret_cast<const uint32_t*>(w);
        const uint32_t b = (row * (cols >> 5) + (u >> 2)) * 6u;
        const uint32_t h = (u >> 1) & 1u, part = u & 1u;
        const float2 dm = unpack_h2(W[b]);
        const uint32_t qh = W[b + 1u], hb = h * 16u + part * 8u;
        a0 = f4_scale_add(f4_add(nib4(W[b + 2u + part * 2u], h * 4u), hi4(qh, hb)), dm.x, dm.y);
        a1 = f4_scale_add(f4_add(nib4(W[b + 3u + part * 2u], h * 4u), hi4(qh, hb + 4u)), dm.x, dm.y);
    } else {
        constexpr uint32_t WORDS = (C == Codec::Q5K) ? 44u : 36u;
        constexpr uint32_t QS = (C == Codec::Q5K) ? 12u : 4u;
        const uint32_t* W = reinterpret_cast<const uint32_t*>(w);
        const uint32_t b = (row * (cols >> 8) + (u >> 5)) * WORDS;
        const uint32_t j = (u >> 2) & 7u;
        const uint32_t half_ = ((u >> 1) & 1u) * 4u + (u & 1u) * 2u;
        const float2 dd = unpack_h2(W[b]);
        const uint2 sm = scale_min(W[b + 1u], W[b + 2u], W[b + 3u], j);
        const float A = dd.x * (float)sm.x, O = -dd.y * (float)sm.y;
        const uint32_t q = b + QS + (j >> 1) * 8u + half_, sh = (j & 1u) * 4u;
        a0 = nib4(W[q], sh);
        a1 = nib4(W[q + 1u], sh);
        if constexpr (C == Codec::Q5K) {
            const uint32_t h0 = W[b + 4u + half_], h1 = W[b + 5u + half_];
            a0.x += (float)((h0 >> j) & 1u) * 16.0f;
            a0.y += (float)((h0 >> (8u + j)) & 1u) * 16.0f;
            a0.z += (float)((h0 >> (16u + j)) & 1u) * 16.0f;
            a0.w += (float)((h0 >> (24u + j)) & 1u) * 16.0f;
            a1.x += (float)((h1 >> j) & 1u) * 16.0f;
            a1.y += (float)((h1 >> (8u + j)) & 1u) * 16.0f;
            a1.z += (float)((h1 >> (16u + j)) & 1u) * 16.0f;
            a1.w += (float)((h1 >> (24u + j)) & 1u) * 16.0f;
        }
        a0 = f4_scale_add(a0, A, O);
        a1 = f4_scale_add(a1, A, O);
    }
}

// =============================================================================================
// Matvec / small-batch kernel (matches shaders/q4_matmul.comp)
// =============================================================================================
template <Codec C, bool Expert, bool SwiGLU>
__global__ void q4_matmul_kernel(uint64_t a0_ptr, uint64_t a1_ptr, uint64_t a2_ptr, uint64_t a3_ptr,
                                 uint64_t x_ptr, uint64_t y_ptr, uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3,
                                 uint32_t rows, uint32_t cols, uint32_t n_or_xmode, uint32_t x_stride, uint32_t y_stride,
                                 uint32_t post, float post_scale) {
    const uint32_t team = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + team;
    if (row >= rows) return;
    const uint32_t chunks = cols / 16u;

    uint64_t w1, s1, w2, s2;
    uint32_t nt, out0;
    uint32_t xrow[NT];

    if constexpr (Expert) {
        const ExpertEntry en = reinterpret_cast<const ExpertEntry*>(a0_ptr)[blockIdx.y];
        nt = en.count;
        out0 = en.start;
        w1 = en.base + p0;
        s1 = en.base + p1;
        w2 = en.base + p2;
        s2 = en.base + p3;
        const uint32_t* toks = reinterpret_cast<const uint32_t*>(a1_ptr);
#pragma unroll
        for (uint32_t t = 0; t < NT; t++)
            xrow[t] = t < nt ? (n_or_xmode == 0u ? toks[en.start + t] : en.start + t) : 0u;
    } else {
        const uint32_t t0 = blockIdx.y * NT;
        nt = min((uint32_t)NT, n_or_xmode - t0);
        out0 = t0;
        w1 = a0_ptr;
        s1 = a1_ptr;
        w2 = a2_ptr;
        s2 = a3_ptr;
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) xrow[t] = t0 + t;
    }

    float acc1[NT], acc2[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        acc1[t] = 0.0f;
        acc2[t] = 0.0f;
    }

    for (uint32_t c = lane; c < chunks; c += 32u) {
        float4 a0, a1, a2, a3;
        decode8<C>(w1, s1, cols, row, c * 2u, a0, a1);
        decode8<C>(w1, s1, cols, row, c * 2u + 1u, a2, a3);
        float4 b0, b1, b2, b3;
        if constexpr (SwiGLU) {
            decode8<C>(w2, s2, cols, row, c * 2u, b0, b1);
            decode8<C>(w2, s2, cols, row, c * 2u + 1u, b2, b3);
        }
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            if (t >= nt) break;
            const float4* xs = reinterpret_cast<const float4*>(x_ptr + (uint64_t)xrow[t] * x_stride * 4u);
            const float4 x0 = xs[c * 4u], x1 = xs[c * 4u + 1u], x2 = xs[c * 4u + 2u], x3 = xs[c * 4u + 3u];
            acc1[t] += f4_dot(a0, x0) + f4_dot(a1, x1) + f4_dot(a2, x2) + f4_dot(a3, x3);
            if constexpr (SwiGLU) {
                acc2[t] += f4_dot(b0, x0) + f4_dot(b1, x1) + f4_dot(b2, x2) + f4_dot(b3, x3);
            }
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        if (t >= nt) break;
        float r = warp_sum(acc1[t]);
        if constexpr (SwiGLU) r = d_silu(r) * warp_sum(acc2[t]);
        if (lane == 0u) {
            if constexpr (!Expert) {
                if (post == 1u) r = d_silu(r * post_scale);
            }
            reinterpret_cast<float*>(y_ptr + (uint64_t)(out0 + t) * y_stride * 4u)[row] = r;
        }
    }
}

// =============================================================================================
// On-the-fly Q8_1 activation quantization + __dp4a int8 matvec (for Codec::Q8, n <= NT)
// =============================================================================================
__global__ void quantize_x_q8_kernel(const float* x, int8_t* xq, float* xs, uint32_t cols, uint32_t x_stride) {
    // 1 warp (32 lanes) per 32-element block; grid = (cols / 32, n)
    const uint32_t blk = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const float v = x[(uint64_t)t * x_stride + blk * 32u + lane];
    const float amax = warp_max(fabsf(v));
    const float sc = amax > 0.0f ? amax / 127.0f : 0.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    xq[(uint64_t)t * cols + blk * 32u + lane] = (int8_t)__float2int_rn(v * inv);
    if (lane == 0u) xs[(uint64_t)t * (cols >> 5) + blk] = sc;
}

template <bool Expert, bool SwiGLU>
__global__ void q4_matmul_q8_dp4a_kernel(uint64_t a0_ptr, uint64_t a1_ptr, uint64_t a2_ptr, uint64_t a3_ptr,
                                         const int8_t* xq, const float* xs, uint64_t y_ptr, uint32_t p0, uint32_t p1,
                                         uint32_t p2, uint32_t p3, uint32_t rows, uint32_t cols, uint32_t n_or_xmode,
                                         uint32_t y_stride, uint32_t post, float post_scale) {
    // 1 warp (32 lanes) per row, 8 rows per block (256 threads). Each lane processes 32-element Q8 blocks
    // (strided by 32 blocks = 1024 elements) using 8x __dp4a per block (4x int4 loads).
    const uint32_t team = threadIdx.x >> 5;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * 8u + team;
    if (row >= rows) return;
    const uint32_t nblk = cols >> 5;

    uint64_t w1, s1, w2, s2;
    uint32_t nt, out0;
    uint32_t xrow[NT];

    if constexpr (Expert) {
        const ExpertEntry en = reinterpret_cast<const ExpertEntry*>(a0_ptr)[blockIdx.y];
        nt = en.count;
        out0 = en.start;
        w1 = en.base + p0;
        s1 = en.base + p1;
        w2 = en.base + p2;
        s2 = en.base + p3;
        const uint32_t* toks = reinterpret_cast<const uint32_t*>(a1_ptr);
#pragma unroll
        for (uint32_t t = 0; t < NT; t++)
            xrow[t] = t < nt ? (n_or_xmode == 0u ? toks[en.start + t] : en.start + t) : 0u;
    } else {
        const uint32_t t0 = blockIdx.y * NT;
        nt = min((uint32_t)NT, n_or_xmode - t0);
        out0 = t0;
        w1 = a0_ptr;
        s1 = a1_ptr;
        w2 = a2_ptr;
        s2 = a3_ptr;
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) xrow[t] = t0 + t;
    }

    const int4* W1 = reinterpret_cast<const int4*>(w1) + (uint64_t)row * (cols >> 4);
    const float* S1 = reinterpret_cast<const float*>(s1) + (uint64_t)row * nblk;
    const int4* W2 = SwiGLU ? (reinterpret_cast<const int4*>(w2) + (uint64_t)row * (cols >> 4)) : nullptr;
    const float* S2 = SwiGLU ? (reinterpret_cast<const float*>(s2) + (uint64_t)row * nblk) : nullptr;

    float acc1[NT], acc2[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        acc1[t] = 0.0f;
        acc2[t] = 0.0f;
    }

    for (uint32_t b = lane; b < nblk; b += 32u) {
        const int4 w1a = W1[b * 2u], w1b = W1[b * 2u + 1u];
        const float sc1 = S1[b];
        int4 w2a, w2b;
        float sc2 = 0.0f;
        if constexpr (SwiGLU) {
            w2a = W2[b * 2u];
            w2b = W2[b * 2u + 1u];
            sc2 = S2[b];
        }
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            if (t >= nt) break;
            const int4* X = reinterpret_cast<const int4*>(xq + (uint64_t)xrow[t] * cols);
            const int4 xa = X[b * 2u], xb = X[b * 2u + 1u];
            const float xsc = xs[(uint64_t)xrow[t] * nblk + b];
            int s = 0;
            s = __dp4a(w1a.x, xa.x, s);
            s = __dp4a(w1a.y, xa.y, s);
            s = __dp4a(w1a.z, xa.z, s);
            s = __dp4a(w1a.w, xa.w, s);
            s = __dp4a(w1b.x, xb.x, s);
            s = __dp4a(w1b.y, xb.y, s);
            s = __dp4a(w1b.z, xb.z, s);
            s = __dp4a(w1b.w, xb.w, s);
            acc1[t] = fmaf((float)s, sc1 * xsc, acc1[t]);
            if constexpr (SwiGLU) {
                int s2i = 0;
                s2i = __dp4a(w2a.x, xa.x, s2i);
                s2i = __dp4a(w2a.y, xa.y, s2i);
                s2i = __dp4a(w2a.z, xa.z, s2i);
                s2i = __dp4a(w2a.w, xa.w, s2i);
                s2i = __dp4a(w2b.x, xb.x, s2i);
                s2i = __dp4a(w2b.y, xb.y, s2i);
                s2i = __dp4a(w2b.z, xb.z, s2i);
                s2i = __dp4a(w2b.w, xb.w, s2i);
                acc2[t] = fmaf((float)s2i, sc2 * xsc, acc2[t]);
            }
        }
    }

#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        if (t >= nt) break;
        float r = warp_sum(acc1[t]);
        if constexpr (SwiGLU) r = d_silu(r) * warp_sum(acc2[t]);
        if (lane == 0u) {
            if constexpr (!Expert) {
                if (post == 1u) r = d_silu(r * post_scale);
            }
            reinterpret_cast<float*>(y_ptr + (uint64_t)(out0 + t) * y_stride * 4u)[row] = r;
        }
    }
}

template <Codec C>
__global__ void q4_matmul_wide_kernel(uint64_t w1, uint64_t s1, uint64_t x_ptr, uint64_t y_ptr, uint32_t rows,
                                      uint32_t cols, uint32_t n, uint32_t x_stride, uint32_t y_stride, uint32_t post,
                                      float post_scale) {
    __shared__ float red[NT][8];
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u;
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t chunks = cols / 16u;
    const uint32_t t0 = blockIdx.y * NT;
    const uint32_t nt = min((uint32_t)NT, n - t0);

    float acc[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.0f;

    for (uint32_t c = tid; c < chunks; c += 256u) {
        float4 a0, a1, a2, a3;
        decode8<C>(w1, s1, cols, row, c * 2u, a0, a1);
        decode8<C>(w1, s1, cols, row, c * 2u + 1u, a2, a3);
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            if (t >= nt) break;
            const float4* xs = reinterpret_cast<const float4*>(x_ptr + (uint64_t)(t0 + t) * x_stride * 4u);
            const float4 x0 = xs[c * 4u], x1 = xs[c * 4u + 1u], x2 = xs[c * 4u + 2u], x3 = xs[c * 4u + 3u];
            acc[t] += f4_dot(a0, x0) + f4_dot(a1, x1) + f4_dot(a2, x2) + f4_dot(a3, x3);
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        if (t >= nt) break;
        const float r = warp_sum(acc[t]);
        if (lane == 0u) red[t][sg] = r;
    }
    __syncthreads();
    if (sg == 0u && lane < nt) {
        float tot = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) tot += red[lane][i];
        if (post == 1u) tot = d_silu(tot * post_scale);
        reinterpret_cast<float*>(y_ptr + (uint64_t)(t0 + lane) * y_stride * 4u)[row] = tot;
    }
}

__global__ void q4_matmul_wide_q8_dp4a_kernel(uint64_t w1, uint64_t s1, const int8_t* xq, const float* xs,
                                              uint64_t y_ptr, uint32_t rows, uint32_t cols, uint32_t nt,
                                              uint32_t y_stride, uint32_t post, float post_scale) {
    __shared__ float red[NT][8];
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u;
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const uint32_t nblk = cols >> 5;
    const int4* W1 = reinterpret_cast<const int4*>(w1) + (uint64_t)row * (cols >> 4);
    const float* S1 = reinterpret_cast<const float*>(s1) + (uint64_t)row * nblk;

    float acc[NT];
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) acc[t] = 0.0f;

    for (uint32_t b = tid; b < nblk; b += 256u) {
        const int4 w1a = W1[b * 2u], w1b = W1[b * 2u + 1u];
        const float sc1 = S1[b];
#pragma unroll
        for (uint32_t t = 0; t < NT; t++) {
            if (t >= nt) break;
            const int4* X = reinterpret_cast<const int4*>(xq + (uint64_t)t * cols);
            const int4 xa = X[b * 2u], xb = X[b * 2u + 1u];
            const float xsc = xs[(uint64_t)t * nblk + b];
            int s = 0;
            s = __dp4a(w1a.x, xa.x, s); s = __dp4a(w1a.y, xa.y, s);
            s = __dp4a(w1a.z, xa.z, s); s = __dp4a(w1a.w, xa.w, s);
            s = __dp4a(w1b.x, xb.x, s); s = __dp4a(w1b.y, xb.y, s);
            s = __dp4a(w1b.z, xb.z, s); s = __dp4a(w1b.w, xb.w, s);
            acc[t] = fmaf((float)s, sc1 * xsc, acc[t]);
        }
    }
#pragma unroll
    for (uint32_t t = 0; t < NT; t++) {
        if (t >= nt) break;
        const float r = warp_sum(acc[t]);
        if (lane == 0u) red[t][sg] = r;
    }
    __syncthreads();
    if (sg == 0u && lane < nt) {
        float tot = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) tot += red[lane][i];
        if (post == 1u) tot = d_silu(tot * post_scale);
        reinterpret_cast<float*>(y_ptr + (uint64_t)lane * y_stride * 4u)[row] = tot;
    }
}

// =============================================================================================
// Tensor Core WMMA GEMM kernel (64x64x32, 8 warps = 256 threads, pure-FP16 smem, 2 blocks/SM)
// =============================================================================================
template <Codec C, bool Expert, bool SwiGLU>
__global__ void q4_gemm_wmma_kernel(uint64_t a0_ptr, uint64_t a1_ptr, uint64_t a2_ptr, uint64_t a3_ptr,
                                    uint64_t x_ptr, uint64_t y_ptr, uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3,
                                    uint32_t rows, uint32_t cols, uint32_t n_or_xmode, uint32_t x_stride,
                                    uint32_t y_stride, uint32_t post, float post_scale) {
    using namespace nvcuda;
    constexpr uint32_t BM = 64, BN = 64, BK = 32, LDA = BK + 8, LDC = BM + 4;
    __shared__ __half Ws[BM * LDA];
    __shared__ __half Ws2[SwiGLU ? BM * LDA : 1];
    __shared__ __half Xs[BN * LDA];
    __shared__ float Cs[32 * LDC];  // 18.5 KB dense / 23.5 KB SWIGLU -> 2 blocks/SM in 48 KB

    const uint32_t tid = threadIdx.x, sg = tid >> 5;
    const uint32_t r0 = blockIdx.x * BM;

    uint64_t w1, s1, w2, s2;
    uint32_t ntok, out0;
    const uint32_t lr = tid / 4u, lk = (tid % 4u) * 8u;
    uint32_t xrow;
    bool tok_ok;

    if constexpr (Expert) {
        const ExpertEntry en = reinterpret_cast<const ExpertEntry*>(a0_ptr)[blockIdx.y];
        ntok = en.count;
        out0 = en.start;
        w1 = en.base + p0;
        s1 = en.base + p1;
        w2 = en.base + p2;
        s2 = en.base + p3;
        tok_ok = lr < ntok;
        const uint32_t* toks = reinterpret_cast<const uint32_t*>(a1_ptr);
        xrow = tok_ok ? (n_or_xmode == 0u ? toks[en.start + lr] : en.start + lr) : 0u;
    } else {
        const uint32_t t0 = blockIdx.y * BN;
        ntok = min(BN, n_or_xmode - t0);
        out0 = t0;
        w1 = a0_ptr;
        s1 = a1_ptr;
        w2 = a2_ptr;
        s2 = a3_ptr;
        tok_ok = lr < ntok;
        xrow = t0 + lr;
    }

    const uint32_t wrow = r0 + lr;
    const float4* xs = reinterpret_cast<const float4*>(x_ptr + (uint64_t)xrow * x_stride * 4u);

    const uint32_t rb = sg >> 1, tb_half = sg & 1u, tb0 = tb_half * 2u;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2], acc2[2];
#pragma unroll
    for (int j = 0; j < 2; j++) {
        wmma::fill_fragment(acc[j], 0.0f);
        if constexpr (SwiGLU) wmma::fill_fragment(acc2[j], 0.0f);
    }

    auto store_h8 = [](__half* dst, float4 a, float4 b) {
        dst[0] = __float2half(a.x); dst[1] = __float2half(a.y); dst[2] = __float2half(a.z); dst[3] = __float2half(a.w);
        dst[4] = __float2half(b.x); dst[5] = __float2half(b.y); dst[6] = __float2half(b.z); dst[7] = __float2half(b.w);
    };

    for (uint32_t k0 = 0; k0 < cols; k0 += BK) {
        float4 a, b;
        decode8<C>(w1, s1, cols, wrow, (k0 + lk) >> 3, a, b);
        store_h8(&Ws[lr * LDA + lk], a, b);
        if constexpr (SwiGLU) {
            float4 a2, b2;
            decode8<C>(w2, s2, cols, wrow, (k0 + lk) >> 3, a2, b2);
            store_h8(&Ws2[lr * LDA + lk], a2, b2);
        }
        float4 x0 = make_float4(0, 0, 0, 0), x1 = make_float4(0, 0, 0, 0);
        if (tok_ok) {
            x0 = xs[(k0 + lk) / 4u];
            x1 = xs[(k0 + lk) / 4u + 1u];
        }
        store_h8(&Xs[lr * LDA + lk], x0, x1);
        __syncthreads();

#pragma unroll
        for (uint32_t kk = 0; kk < BK; kk += 16u) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa, fa2;
            wmma::load_matrix_sync(fa, &Ws[rb * 16u * LDA + kk], LDA);
            if constexpr (SwiGLU) wmma::load_matrix_sync(fa2, &Ws2[rb * 16u * LDA + kk], LDA);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, &Xs[(tb0 + j) * 16u * LDA + kk], LDA);
                wmma::mma_sync(acc[j], fa, fb, acc[j]);
                if constexpr (SwiGLU) wmma::mma_sync(acc2[j], fa2, fb, acc2[j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (uint32_t j = 0; j < 2u; j++) {
        if constexpr (SwiGLU) {
            for (int i = 0; i < acc[j].num_elements; i++) acc[j].x[i] = d_silu(acc[j].x[i]) * acc2[j].x[i];
        }
        wmma::store_matrix_sync(&Cs[tb_half * 16u * LDC + rb * 16u], acc[j], LDC, wmma::mem_col_major);
        __syncthreads();
        for (uint32_t i = tid; i < BM * 32u; i += 256u) {
            const uint32_t lt = i / BM, r = i % BM;
            const uint32_t t = (lt < 16u) ? (lt + j * 16u) : (32u + (lt - 16u) + j * 16u);
            if (t < ntok) {
                float v = Cs[lt * LDC + r];
                if constexpr (!Expert) {
                    if (post == 1u) v = d_silu(v * post_scale);
                }
                reinterpret_cast<float*>(y_ptr + (uint64_t)(out0 + t) * y_stride * 4u)[r0 + r] = v;
            }
        }
        __syncthreads();
    }
}

// =============================================================================================
// Tiled GEMM kernel (matches shaders/q4_gemm.comp: 64x64x32, 256 threads)
// =============================================================================================
template <Codec C, bool Expert, bool SwiGLU>
__global__ void q4_gemm_kernel(uint64_t a0_ptr, uint64_t a1_ptr, uint64_t a2_ptr, uint64_t a3_ptr,
                               uint64_t x_ptr, uint64_t y_ptr, uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3,
                               uint32_t rows, uint32_t cols, uint32_t n_or_xmode, uint32_t x_stride, uint32_t y_stride,
                               uint32_t post, float post_scale) {
    constexpr uint32_t BM = 64, BN = 64, BK = 32;
    __shared__ float Ws[BK][BM];
    __shared__ float Ws2[SwiGLU ? BK : 1][SwiGLU ? BM : 1];
    __shared__ float Xs[BK][BN];

    const uint32_t tid = threadIdx.x, ty = tid / 16u, tx = tid % 16u;
    const uint32_t r0 = blockIdx.x * BM;

    uint64_t w1, s1, w2, s2;
    uint32_t ntok, out0;
    const uint32_t lr = tid / 4u, lk = (tid % 4u) * 8u;
    uint32_t xrow;
    bool tok_ok;

    if constexpr (Expert) {
        const ExpertEntry en = reinterpret_cast<const ExpertEntry*>(a0_ptr)[blockIdx.y];
        ntok = en.count;
        out0 = en.start;
        w1 = en.base + p0;
        s1 = en.base + p1;
        w2 = en.base + p2;
        s2 = en.base + p3;
        tok_ok = lr < ntok;
        const uint32_t* toks = reinterpret_cast<const uint32_t*>(a1_ptr);
        xrow = tok_ok ? (n_or_xmode == 0u ? toks[en.start + lr] : en.start + lr) : 0u;
    } else {
        const uint32_t t0 = blockIdx.y * BN;
        ntok = min(BN, n_or_xmode - t0);
        out0 = t0;
        w1 = a0_ptr;
        s1 = a1_ptr;
        w2 = a2_ptr;
        s2 = a3_ptr;
        tok_ok = lr < ntok;
        xrow = t0 + lr;
    }

    const uint32_t wrow = r0 + lr;
    const float4* xs = reinterpret_cast<const float4*>(x_ptr + (uint64_t)xrow * x_stride * 4u);

    float4 acc[4], acc2[4];
#pragma unroll
    for (uint32_t i = 0; i < 4u; i++) {
        acc[i] = make_float4(0, 0, 0, 0);
        acc2[i] = make_float4(0, 0, 0, 0);
    }

    for (uint32_t k0 = 0; k0 < cols; k0 += BK) {
        float4 a, b;
        decode8<C>(w1, s1, cols, wrow, (k0 + lk) >> 3, a, b);
        Ws[lk + 0u][lr] = a.x; Ws[lk + 1u][lr] = a.y; Ws[lk + 2u][lr] = a.z; Ws[lk + 3u][lr] = a.w;
        Ws[lk + 4u][lr] = b.x; Ws[lk + 5u][lr] = b.y; Ws[lk + 6u][lr] = b.z; Ws[lk + 7u][lr] = b.w;
        if constexpr (SwiGLU) {
            float4 a2, b2;
            decode8<C>(w2, s2, cols, wrow, (k0 + lk) >> 3, a2, b2);
            Ws2[lk + 0u][lr] = a2.x; Ws2[lk + 1u][lr] = a2.y; Ws2[lk + 2u][lr] = a2.z; Ws2[lk + 3u][lr] = a2.w;
            Ws2[lk + 4u][lr] = b2.x; Ws2[lk + 5u][lr] = b2.y; Ws2[lk + 6u][lr] = b2.z; Ws2[lk + 7u][lr] = b2.w;
        }
        float4 x0 = make_float4(0, 0, 0, 0), x1 = make_float4(0, 0, 0, 0);
        if (tok_ok) {
            x0 = xs[(k0 + lk) / 4u];
            x1 = xs[(k0 + lk) / 4u + 1u];
        }
        Xs[lk + 0u][lr] = x0.x; Xs[lk + 1u][lr] = x0.y; Xs[lk + 2u][lr] = x0.z; Xs[lk + 3u][lr] = x0.w;
        Xs[lk + 4u][lr] = x1.x; Xs[lk + 5u][lr] = x1.y; Xs[lk + 6u][lr] = x1.z; Xs[lk + 7u][lr] = x1.w;
        __syncthreads();

#pragma unroll
        for (uint32_t k = 0; k < BK; k++) {
            const float4 av = reinterpret_cast<const float4*>(&Ws[k][ty * 4u])[0];
            const float4 bv = reinterpret_cast<const float4*>(&Xs[k][tx * 4u])[0];
            acc[0] = f4_mad(bv, av.x, acc[0]);
            acc[1] = f4_mad(bv, av.y, acc[1]);
            acc[2] = f4_mad(bv, av.z, acc[2]);
            acc[3] = f4_mad(bv, av.w, acc[3]);
            if constexpr (SwiGLU) {
                const float4 av2 = reinterpret_cast<const float4*>(&Ws2[k][ty * 4u])[0];
                acc2[0] = f4_mad(bv, av2.x, acc2[0]);
                acc2[1] = f4_mad(bv, av2.y, acc2[1]);
                acc2[2] = f4_mad(bv, av2.z, acc2[2]);
                acc2[3] = f4_mad(bv, av2.w, acc2[3]);
            }
        }
        __syncthreads();
    }

    auto lane_val = [](float4 v, uint32_t j) -> float { return j == 0 ? v.x : (j == 1 ? v.y : (j == 2 ? v.z : v.w)); };
#pragma unroll
    for (uint32_t j = 0; j < 4u; j++) {
        const uint32_t tok = tx * 4u + j;
        if (tok >= ntok) break;
        float* y = reinterpret_cast<float*>(y_ptr + (uint64_t)(out0 + tok) * y_stride * 4u);
#pragma unroll
        for (uint32_t i = 0; i < 4u; i++) {
            float r = lane_val(acc[i], j);
            if constexpr (SwiGLU) r = d_silu(r) * lane_val(acc2[i], j);
            if constexpr (!Expert) {
                if (post == 1u) r = d_silu(r * post_scale);
            }
            y[r0 + ty * 4u + i] = r;
        }
    }
}

// =============================================================================================
// Hyper-connections, PLE, RMSNorm, Conv1d, DeltaNet, Attention, MoE kernels
// =============================================================================================
__global__ void q4_hc_norm_kernel(const float* x, const float* w, float* out, uint32_t dim, uint32_t hc, float eps) {
    __shared__ float red[8];
    const uint32_t t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const float* xr = x + (uint64_t)(t * hc + s) * dim;
    float ss = 0.0f;
    for (uint32_t i = tid; i < dim; i += 256u) ss += xr[i] * xr[i];
    ss = warp_sum(ss);
    if ((tid & 31u) == 0u) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (uint32_t i = 0; i < 8u; i++) tot += red[i];
    const float r = rsqrtf(tot / (float)dim + eps);
    float* o = out + (uint64_t)(t * hc + s) * dim;
    const float* wr = w + (uint64_t)s * dim;
    for (uint32_t i = tid; i < dim; i += 256u) o[i] = xr[i] * r * wr[i];
}

__global__ void q4_hc_mix_kernel(const float* xnorm, const float* gate, float* out, uint32_t dim, uint32_t hc) {
    const uint32_t d = blockIdx.x * 256u + threadIdx.x, t = blockIdx.y;
    if (d >= dim) return;
    const float* xs = xnorm + (uint64_t)t * hc * dim;
    const float* g = gate + (uint64_t)t * hc * dim;
    float acc = 0.0f;
    for (uint32_t s = 0; s < hc; s++) acc += xs[s * dim + d] * d_sigmoid(g[s * dim + d]);
    out[(uint64_t)t * dim + d] = acc / (float)hc;
}

__global__ void q4_hc_combine_kernel(float* x, const float* inj, const float* blk, uint32_t dim, uint32_t hc) {
    const uint32_t d = blockIdx.x * 256u + threadIdx.x, t = blockIdx.y;
    if (d >= dim) return;
    float* xr = x + (uint64_t)t * hc * dim;
    const float bv = blk[(uint64_t)t * dim + d];
    for (uint32_t s = 0; s < hc; s++)
        xr[s * dim + d] += 2.0f * d_sigmoid(inj[t * hc + s] / (float)hc) * bv;
}

__global__ void q4_hc_cat_kernel(const float* e, const float* h, float* out, uint32_t dim, uint32_t hc) {
    const uint32_t d = blockIdx.x * 256u + threadIdx.x, r = blockIdx.y;
    if (d >= 2u * dim) return;
    out[(uint64_t)r * 2u * dim + d] = d < dim ? e[(uint64_t)(r / hc) * dim + d] : h[(uint64_t)r * dim + (d - dim)];
}

__global__ void add_rmsnorm_kernel(float* x, const float* delta, const float* w, float* out, uint32_t dim, float eps) {
    __shared__ float red[8];
    const uint32_t t = blockIdx.x, tid = threadIdx.x;
    float* xr = x + (uint64_t)t * dim;
    float ss = 0.0f;
    for (uint32_t i = tid; i < dim; i += 256u) {
        float v = xr[i];
        if (delta) {
            v += delta[(uint64_t)t * dim + i];
            xr[i] = v;
        }
        ss += v * v;
    }
    ss = warp_sum(ss);
    if ((tid & 31u) == 0u) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (uint32_t i = 0; i < 8u; i++) tot += red[i];
    const float r = rsqrtf(tot / (float)dim + eps);
    float* o = out + (uint64_t)t * dim;
    for (uint32_t i = tid; i < dim; i += 256u) o[i] = xr[i] * r * (1.0f + w[i]);
}

__global__ void q4_ple_gate_kernel(const float* key_b, const float* val_b, const float* x_b, const float* wk_b,
                                   const float* wq_b, const float* wc_b, float* gated_b, float* pad_b, uint32_t D,
                                   uint32_t hc, uint32_t hist, float eps) {
    __shared__ float red[3][8];
    auto wg_sum = [&](float v, uint32_t slot) {
        v = warp_sum(v);
        if ((threadIdx.x & 31u) == 0u) red[slot][threadIdx.x >> 5] = v;
        __syncthreads();
        float tot = 0.0f;
        for (uint32_t i = 0; i < 8u; i++) tot += red[slot][i];
        return tot;
    };
    const uint32_t t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x;
    const uint64_t row = (uint64_t)(t * hc + s) * D, wrow = (uint64_t)s * D;
    const float* key = key_b + row;
    const float* x = x_b + row;
    const float* value = val_b + (uint64_t)t * D;
    const float* wk = wk_b + wrow;
    const float* wq = wq_b + wrow;
    const float* wc = wc_b + wrow;
    float sk = 0.0f, sq = 0.0f;
    for (uint32_t i = tid; i < D; i += 256u) {
        sk += key[i] * key[i];
        sq += x[i] * x[i];
    }
    const float rk = rsqrtf(wg_sum(sk, 0u) / (float)D + eps);
    const float rq = rsqrtf(wg_sum(sq, 1u) / (float)D + eps);
    float dp = 0.0f;
    for (uint32_t i = tid; i < D; i += 256u) dp += key[i] * wk[i] * x[i] * wq[i];
    const float d = wg_sum(dp, 2u) * rk * rq * rsqrtf((float)D);
    const float sgn = d > 0.0f ? 1.0f : (d < 0.0f ? -1.0f : 0.0f);
    const float g = d_sigmoid(sgn * sqrtf(fmaxf(fabsf(d), 1e-6f)));
    __syncthreads();
    float sv = 0.0f;
    for (uint32_t i = tid; i < D; i += 256u) {
        const float v = g * value[i];
        sv += v * v;
    }
    const float rv = rsqrtf(wg_sum(sv, 0u) / (float)D + eps);
    float* gated = gated_b + row;
    float* pad = pad_b + (uint64_t)((hist + t) * hc + s) * D;
    for (uint32_t i = tid; i < D; i += 256u) {
        const float v = g * value[i];
        gated[i] = v;
        pad[i] = v * rv * wc[i];
    }
}

__global__ void q4_ple_conv_kernel(float* x, const float* gated, const float* pad, const float* w, uint32_t width,
                                   uint32_t kernel, uint32_t dil, uint32_t hist) {
    const uint32_t ch = blockIdx.x * 256u + threadIdx.x, t = blockIdx.y;
    if (ch >= width) return;
    float acc = 0.0f;
    for (uint32_t k = 0; k < kernel; k++)
        acc += w[ch * kernel + k] * pad[(uint64_t)(hist + t - (kernel - 1u - k) * dil) * width + ch];
    const uint64_t i = (uint64_t)t * width + ch;
    x[i] += gated[i] + d_silu(acc);
}

template <bool Snap>
__global__ void conv1d_kernel(float* data, const float* w, float* state, uint32_t n, uint32_t channels, uint64_t snap,
                              uint32_t snap_stride) {
    constexpr uint32_t K = 4;
    const uint32_t ch = blockIdx.x * 256u + threadIdx.x;
    if (ch >= channels) return;
    float wk[K], hist[K - 1];
#pragma unroll
    for (uint32_t k = 0; k < K; k++) wk[k] = w[ch * K + k];
#pragma unroll
    for (uint32_t k = 0; k < K - 1; k++) hist[k] = state[k * channels + ch];
    for (uint32_t t = 0; t < n; t++) {
        const uint32_t idx = t * channels + ch;
        const float in_ = data[idx];
        float acc = wk[K - 1] * in_;
#pragma unroll
        for (uint32_t k = 0; k < K - 1; k++) acc += wk[k] * hist[k];
#pragma unroll
        for (uint32_t k = 0; k < K - 2; k++) hist[k] = hist[k + 1];
        hist[K - 2] = in_;
        data[idx] = d_silu(acc);
        if constexpr (Snap) {
            if (t + 1u < n) {
                float* sn = reinterpret_cast<float*>(snap + (uint64_t)t * snap_stride);
#pragma unroll
                for (uint32_t k = 0; k < K - 1; k++) sn[k * channels + ch] = hist[k];
            }
        }
    }
#pragma unroll
    for (uint32_t k = 0; k < K - 1; k++) state[k * channels + ch] = hist[k];
}

template <bool Snap>
__global__ void deltanet_tiled_kernel(const float* qkv, const float* a, const float* b, const float* neg_a,
                                      const float* dt_bias, float* state, float* out_, uint32_t n, uint32_t conv_dim,
                                      uint32_t key_dim, uint32_t n_v_heads, uint32_t ratio, uint64_t snap,
                                      uint32_t snap_stride) {
    constexpr uint32_t KD = 128, DV = 128, VS = 16, KG = 8, KPER = KD / KG;
    __shared__ float qs[KD], ks[KD], part[KG][VS], dsh[VS], red[8];

    const uint32_t h = blockIdx.x, vs = blockIdx.y, tid = threadIdx.x;
    const uint32_t col = tid % VS, kg = tid / VS;
    const uint32_t kh = h % (n_v_heads / ratio), vcol = vs * VS + col;
    float* st = state + (uint64_t)h * KD * DV;
    float S[KPER];
#pragma unroll
    for (uint32_t i = 0; i < KPER; i++) S[i] = st[(kg * KPER + i) * DV + vcol];
    const float negA = neg_a[h], dtb = dt_bias[h];
    const float qscale = rsqrtf((float)KD);

    for (uint32_t t = 0; t < n; t++) {
        const float* row = qkv + (uint64_t)t * conv_dim;
        const float qv = row[kh * KD + tid], kv_ = row[key_dim + kh * KD + tid];
        const float sq = warp_sum(qv * qv), sk = warp_sum(kv_ * kv_);
        if ((tid & 31u) == 0u) {
            red[tid >> 5] = sq;
            red[4u + (tid >> 5)] = sk;
        }
        __syncthreads();
        float nq = 0.0f, nk = 0.0f;
        for (uint32_t i = 0; i < 4u; i++) {
            nq += red[i];
            nk += red[4u + i];
        }
        qs[tid] = qv * rsqrtf(nq + 1e-6f) * qscale;
        ks[tid] = kv_ * rsqrtf(nk + 1e-6f);
        __syncthreads();

        const float beta = d_sigmoid(b[t * n_v_heads + h]);
        const float decay = expf(negA * d_softplus(a[t * n_v_heads + h] + dtb));
        float p = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < KPER; i++) {
            S[i] *= decay;
            p += S[i] * ks[kg * KPER + i];
        }
        part[kg][col] = p;
        __syncthreads();
        if (kg == 0u) {
            float kvm = 0.0f;
#pragma unroll
            for (uint32_t g = 0; g < KG; g++) kvm += part[g][col];
            dsh[col] = (row[2u * key_dim + h * DV + vcol] - kvm) * beta;
        }
        __syncthreads();
        const float d = dsh[col];
        p = 0.0f;
#pragma unroll
        for (uint32_t i = 0; i < KPER; i++) {
            S[i] += ks[kg * KPER + i] * d;
            p += S[i] * qs[kg * KPER + i];
        }
        part[kg][col] = p;
        __syncthreads();
        if (kg == 0u) {
            float o = 0.0f;
#pragma unroll
            for (uint32_t g = 0; g < KG; g++) o += part[g][col];
            out_[(uint64_t)t * n_v_heads * DV + h * DV + vcol] = o;
        }
        __syncthreads();
        if constexpr (Snap) {
            if (t + 1u < n) {
                float* sn = reinterpret_cast<float*>(snap + (uint64_t)t * snap_stride + (uint64_t)h * KD * DV * 4u);
#pragma unroll
                for (uint32_t i = 0; i < KPER; i++) sn[(kg * KPER + i) * DV + vcol] = S[i];
            }
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < KPER; i++) st[(kg * KPER + i) * DV + vcol] = S[i];
}

__global__ void gated_norm_sigmoid_kernel(float* o, const float* z, const float* w, uint32_t n_heads, float eps) {
    __shared__ float red[4];
    const uint32_t t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const uint32_t idx = (t * n_heads + h) * 128u + tid;
    const float v = o[idx];
    const float ss = warp_sum(v * v);
    if ((tid & 31u) == 0u) red[tid >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (uint32_t i = 0; i < 4u; i++) tot += red[i];
    const float g = d_sigmoid(z[idx]);
    o[idx] = v * rsqrtf(tot / 128.0f + eps) * w[tid] * g;
}

__global__ void qk_norm_rope_kv16_kernel(const float* qg, const float* k, const float* v, float* q_out,
                                         uint32_t* kcache, uint32_t* vcache, const float* qn, const float* kn,
                                         const float* rope, uint32_t pos_val, const AttnDynParams* dyn,
                                         uint32_t n_heads, uint32_t n_kv, uint32_t rope_dim, float eps) {
    constexpr uint32_t D = 256;
    __shared__ float sh[D], sk[D], red[8];
    const uint32_t pos = dyn ? dyn->pos : pos_val;
    const uint32_t t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
    const bool is_q = hh < n_heads;
    const uint32_t h = is_q ? hh : hh - n_heads;
    const float x = is_q ? qg[(t * n_heads + h) * 2u * D + d] : k[(t * n_kv + h) * D + d];
    const float ss = warp_sum(x * x);
    if ((d & 31u) == 0u) red[d >> 5] = ss;
    __syncthreads();
    float tot = 0.0f;
    for (uint32_t i = 0; i < 8u; i++) tot += red[i];
    const float w = (is_q ? qn : kn)[d];
    float y = x * rsqrtf(tot / (float)D + eps) * (1.0f + w);
    sh[d] = y;
    __syncthreads();
    const uint32_t p = pos + t, half_ = rope_dim / 2u;
    if (d < rope_dim) {
        const uint32_t i = d % half_;
        const float* rp = rope + (uint64_t)p * rope_dim;
        const float c = rp[2u * i], s = rp[2u * i + 1u];
        y = d < half_ ? sh[d] * c - sh[d + half_] * s : sh[d] * c + sh[i] * s;
    }
    sk[d] = y;
    __syncthreads();
    if (is_q) {
        q_out[(t * n_heads + h) * D + d] = y;
    } else if ((d & 1u) == 0u) {
        const uint64_t ci = ((uint64_t)(p * n_kv + h) * D + d) / 2u;
        const uint64_t vi = (uint64_t)(t * n_kv + h) * D + d;
        kcache[ci] = pack_h2(y, sk[d + 1u]);
        vcache[ci] = pack_h2(v[vi], v[vi + 1u]);
    }
}

template <bool Qsa>
__global__ void attention_kv16_kernel(const float* q, const uint8_t* kcache, const uint8_t* vcache, float* part,
                                      uint32_t pos_val, uint32_t n_heads, uint32_t n_kv, uint32_t chunk_val,
                                      uint32_t n_split, const AttnDynParams* dyn, float scale,
                                      const uint32_t* mask_base, uint32_t cr, uint32_t mask_words) {
    constexpr uint32_t D = 256, PART = D + 2;
    __shared__ float sm[8], sl[8], sacc[8][D];

    const uint32_t pos = dyn ? dyn->pos : pos_val;
    const uint32_t chunk = dyn ? dyn->chunk : chunk_val;
    const uint32_t t = blockIdx.x, h = blockIdx.y, z = blockIdx.z;
    const uint32_t tid = threadIdx.x, sg = tid >> 5, lane = tid & 31u;
    const uint32_t kvh = h / (n_heads / n_kv), len = pos + t + 1u;
    const uint32_t jb = z * chunk, je = min(len, jb + chunk);
    const float4* q4 = reinterpret_cast<const float4*>(q + (uint64_t)(t * n_heads + h) * D);
    const float4 qa = f4_scale_add(q4[lane * 2u], scale, 0.0f);
    const float4 qb = f4_scale_add(q4[lane * 2u + 1u], scale, 0.0f);

    float m = -1e30f, l = 0.0f;
    float4 oa = make_float4(0, 0, 0, 0), ob = make_float4(0, 0, 0, 0);
    const uint32_t* mask = Qsa ? (mask_base + (uint64_t)t * mask_words) : nullptr;
    const uint32_t tail = Qsa ? ((len / cr) * cr) : 0u;

    for (uint32_t j = jb + sg; j < je; j += 8u) {
        if constexpr (Qsa) {
            const uint32_t blk = j / cr;
            if (j < tail && (mask[blk >> 5] & (1u << (blk & 31u))) == 0u) continue;
        }
        const uint64_t off = (uint64_t)(j * n_kv + kvh) * (D * 2u);
        const uint4 kw = reinterpret_cast<const uint4*>(kcache + off)[lane];
        const uint4 vw = reinterpret_cast<const uint4*>(vcache + off)[lane];
        const float2 k0 = unpack_h2(kw.x), k1 = unpack_h2(kw.y), k2 = unpack_h2(kw.z), k3 = unpack_h2(kw.w);
        const float2 v0 = unpack_h2(vw.x), v1 = unpack_h2(vw.y), v2 = unpack_h2(vw.z), v3 = unpack_h2(vw.w);
        const float4 ka = make_float4(k0.x, k0.y, k1.x, k1.y), kb = make_float4(k2.x, k2.y, k3.x, k3.y);
        const float4 va = make_float4(v0.x, v0.y, v1.x, v1.y), vb = make_float4(v2.x, v2.y, v3.x, v3.y);
        const float s = warp_sum(f4_dot(qa, ka) + f4_dot(qb, kb));
        const float mn = fmaxf(m, s), corr = expf(m - mn), p = expf(s - mn);
        l = l * corr + p;
        oa = f4_mad(va, p, f4_scale_add(oa, corr, 0.0f));
        ob = f4_mad(vb, p, f4_scale_add(ob, corr, 0.0f));
        m = mn;
    }

    if (lane == 0u) {
        sm[sg] = m;
        sl[sg] = l;
    }
    sacc[sg][lane * 8u + 0u] = oa.x; sacc[sg][lane * 8u + 1u] = oa.y;
    sacc[sg][lane * 8u + 2u] = oa.z; sacc[sg][lane * 8u + 3u] = oa.w;
    sacc[sg][lane * 8u + 4u] = ob.x; sacc[sg][lane * 8u + 5u] = ob.y;
    sacc[sg][lane * 8u + 6u] = ob.z; sacc[sg][lane * 8u + 7u] = ob.w;
    __syncthreads();

    float M = -1e30f;
    for (uint32_t i = 0; i < 8u; i++) if (sl[i] > 0.0f) M = fmaxf(M, sm[i]);
    float L = 0.0f, o = 0.0f;
    for (uint32_t i = 0; i < 8u; i++) {
        const float c = sl[i] > 0.0f ? expf(sm[i] - M) : 0.0f;
        L += sl[i] * c;
        o += sacc[i][tid] * c;
    }
    float* pout = part + (uint64_t)((t * n_heads + h) * n_split + z) * PART;
    pout[2u + tid] = o;
    if (tid == 0u) {
        pout[0] = M;
        pout[1] = L;
    }
}

__global__ void attn_combine_kernel(const float* part_base, const float* qg, float* out_, uint32_t n_heads, uint32_t n_split) {
    constexpr uint32_t D = 256, PART = D + 2;
    const uint32_t t = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float* part = part_base + (uint64_t)(t * n_heads + h) * n_split * PART;
    float M = -1e30f;
    for (uint32_t z = 0; z < n_split; z++)
        if (part[z * PART + 1u] > 0.0f) M = fmaxf(M, part[z * PART]);
    float L = 0.0f, o = 0.0f;
    for (uint32_t z = 0; z < n_split; z++) {
        const float l = part[z * PART + 1u];
        if (l <= 0.0f) continue;
        const float c = expf(part[z * PART] - M);
        L += l * c;
        o += part[z * PART + 2u + d] * c;
    }
    const float gate = qg[(t * n_heads + h) * 2u * D + D + d];
    out_[(t * n_heads + h) * D + d] = (o / L) * d_sigmoid(gate);
}

__global__ void moe_route_kernel(const float* logits, uint32_t* sel, float* selw, uint32_t n_experts, uint32_t top_k) {
    constexpr uint32_t PER = 4;
    __shared__ float s_val[8];
    __shared__ uint32_t s_idx[8];

    auto block_max = [&](float v) {
        v = warp_max(v);
        if ((threadIdx.x & 31u) == 0u) s_val[threadIdx.x >> 5] = v;
        __syncthreads();
        float r = s_val[0];
        for (uint32_t i = 1; i < 8u; i++) r = fmaxf(r, s_val[i]);
        __syncthreads();
        return r;
    };
    auto block_sum = [&](float v) {
        v = warp_sum(v);
        if ((threadIdx.x & 31u) == 0u) s_val[threadIdx.x >> 5] = v;
        __syncthreads();
        float r = 0.0f;
        for (uint32_t i = 0; i < 8u; i++) r += s_val[i];
        __syncthreads();
        return r;
    };

    const uint32_t t = blockIdx.x, tid = threadIdx.x, E = n_experts;
    const float* lg = logits + (uint64_t)t * E;
    float p[PER];
    float mx = -1e30f;
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) {
        const uint32_t e = tid + 256u * j;
        p[j] = e < E ? lg[e] : -1e30f;
        mx = fmaxf(mx, p[j]);
    }
    mx = block_max(mx);
    float s = 0.0f;
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) {
        const uint32_t e = tid + 256u * j;
        p[j] = e < E ? expf(p[j] - mx) : 0.0f;
        s += p[j];
    }
    s = block_sum(s);
#pragma unroll
    for (uint32_t j = 0; j < PER; j++) p[j] = (tid + 256u * j) < E ? p[j] / s : -1.0f;

    float chosen_p = 0.0f, total = 0.0f;
    uint32_t chosen_e = 0u;
    for (uint32_t k = 0; k < top_k; k++) {
        float bv = -1.0f;
        uint32_t bi = 0xFFFFFFFFu;
#pragma unroll
        for (uint32_t j = 0; j < PER; j++)
            if (p[j] > bv) {
                bv = p[j];
                bi = tid + 256u * j;
            }
        const float m = warp_max(bv);
        const uint32_t mi = warp_min_u32(bv == m ? bi : 0xFFFFFFFFu);
        if ((tid & 31u) == 0u) {
            s_val[tid >> 5] = m;
            s_idx[tid >> 5] = mi;
        }
        __syncthreads();
        float M = s_val[0];
        uint32_t I = s_idx[0];
        for (uint32_t i = 1; i < 8u; i++)
            if (s_val[i] > M || (s_val[i] == M && s_idx[i] < I)) {
                M = s_val[i];
                I = s_idx[i];
            }
        __syncthreads();
#pragma unroll
        for (uint32_t j = 0; j < PER; j++)
            if (tid + 256u * j == I) p[j] = -1.0f;
        if (tid == k) {
            chosen_p = M;
            chosen_e = I;
        }
        total += M;
    }
    if (tid < top_k) {
        selw[t * top_k + tid] = chosen_p / total;
        sel[t * top_k + tid] = chosen_e;
    }
}

__global__ void moe_combine_hc_kernel(float* x, const float* inj, const float* sgate, const float* sy,
                                      const float* ey, const __half* cy16, const uint32_t* row_ids, const float* selw,
                                      uint32_t dim, uint32_t hc, uint32_t top_k, uint32_t gpu_rows) {
    const uint32_t d = blockIdx.x * 256u + threadIdx.x, t = blockIdx.y;
    if (d >= dim) return;
    float blk = d_sigmoid(sgate[t]) * sy[(uint64_t)t * dim + d];
    const uint32_t base = t * top_k;
    for (uint32_t k = 0; k < top_k; k++) {
        const uint32_t r = row_ids[base + k];
        const float w = selw[base + k];
        const float v = (r < gpu_rows) ? ey[(uint64_t)r * dim + d]
                                       : __half2float(cy16[(uint64_t)(r - gpu_rows) * dim + d]);
        blk += w * v;
    }
    float* xr = x + (uint64_t)t * hc * dim;
    for (uint32_t s = 0; s < hc; s++)
        xr[s * dim + d] += 2.0f * d_sigmoid(inj[t * hc + s] / (float)hc) * blk;
}

// =============================================================================================
// Host helpers & Backend
// =============================================================================================
uint32_t groups(uint32_t n, uint32_t per) { return (n + per - 1) / per; }
size_t align256(size_t v) { return (v + 255) & ~size_t(255); }

void unpack_q8_row(const uint8_t* src, int cols, uint8_t* w, float* s) {
    for (int j = 0; j < cols; j += 32, src += 34) {
        uint16_t h;
        std::memcpy(&h, src, 2);
        s[j / 32] = ck::half_to_f32(h);
        std::memcpy(w + j, src + 2, 32);
    }
}

size_t expert_part_bytes(const gq::GMat& m) {
    if (m.type == GType::Q8_0) return align256((size_t)m.rows * m.cols) + align256((size_t)m.rows * (m.cols / 32) * 4);
    return align256(m.nbytes());
}

struct CM {
    uint64_t w = 0, s = 0;
    int rows = 0, cols = 0;
    bool q8 = false;
};
struct CHc {
    uint64_t norm = 0;
    CM down, up, inject;
    bool has_inject = false;
};

struct CLayer {
    CHc hc_attn, hc_ffn;
    CM q, k, v, o;
    uint64_t qn = 0, kn = 0;
    CM qkv, z, out, alpha, beta;
    uint64_t conv = 0, neg_a = 0, dt_bias = 0, lnorm = 0;
    CM ple_key, ple_value;
    uint64_t ple_wk = 0, ple_wq = 0, ple_wc = 0, ple_conv = 0;
    CM router, sgate, sh_gate, sh_up, sh_down;
    uint32_t off_up = 0, off_down = 0, s_gate = 0, s_up = 0, s_down = 0;
    size_t slot_size = 0;
    int cls = 0;
    Codec cu = Codec::Q4K, cd = Codec::Q8;
};

class CudaArena {
public:
    ~CudaArena() {
        for (void* p : chunks_) cudaFree(p);
    }
    uint64_t alloc(size_t size) {
        size = align256(size);
        if (size > CHUNK) {
            void* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, size));
            chunks_.push_back(p);
            allocated_ += size;
            return reinterpret_cast<uint64_t>(p);
        }
        if (!cur_ || used_ + size > CHUNK) {
            void* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, CHUNK));
            chunks_.push_back(p);
            cur_ = reinterpret_cast<uint8_t*>(p);
            used_ = 0;
            allocated_ += CHUNK;
        }
        uint64_t addr = reinterpret_cast<uint64_t>(cur_ + used_);
        used_ += size;
        return addr;
    }
    size_t allocated() const { return allocated_; }

private:
    std::vector<void*> chunks_;
    uint8_t* cur_ = nullptr;
    size_t used_ = 0, allocated_ = 0;
};

class Qwen4Cuda final : public Backend {
public:
    Qwen4Cuda(const Q4Config& c, const Q4Weights& w, int max_ctx, const VulkanOptions& opt)
        : c_(c), w_(w), max_ctx_(max_ctx), opt_(opt), cpu_(expert_threads(opt.threads)), qsa_(c, w, max_ctx, cpu_) {
        if (c.head_dim != 256 || c.lin_k_dim != 128 || c.lin_v_dim != 128 || c.conv_kernel != 4 || c.hidden % 256 ||
            c.n_experts > 1024 || c.rope_dim > 256)
            throw std::runtime_error("model dimensions not supported by the CUDA kernels");
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);  // low-latency sync on WDDM/Linux
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::fprintf(stderr, "[cuda] %s | SM %d.%d | SMs %d | VRAM %.1f GB\n", prop.name, prop.major, prop.minor,
                     prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0);
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&route_done_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&fill_done_, cudaEventDisableTiming));

        use_dp4a_ = std::getenv("QW_DP4A") != nullptr;
        use_graph_ = std::getenv("QW_GRAPH") != nullptr;
        draft_ = w.has_mtp ? std::max(0, std::min(opt.draft, NT - 1)) : 0;
        if (const char* tp = std::getenv("QW_TRACE")) trace_ = std::fopen(tp, "ab");

        size_t free_b = 0, total_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
        const uint64_t nvml_free = qw_system_free_vram(prop.name);
        sys_free0_ = nvml_free ? std::min<size_t>(free_b, (size_t)nvml_free) : free_b;
        if (const char* lim = std::getenv("QW_VRAM_LIMIT_MB")) {
            const size_t cap = ((size_t)std::max(0ll, std::atoll(lim)) << 20) + VRAM_MARGIN;
            sys_free0_ = sys_free0_ ? std::min(sys_free0_, cap) : cap;
        }

        upload_weights();
        alloc_state();
        setup_experts();
        reset();
        warm_experts();
    }

    ~Qwen4Cuda() override {
        cudaDeviceSynchronize();
        std::fprintf(stderr, "[cuda] expert lookups: %llu on the GPU (cached), %llu on the CPU; %llu cache fills\n",
                     (unsigned long long)hits_, (unsigned long long)misses_, (unsigned long long)fills_);
        if (std::getenv("QW_PROFILE")) {
            std::fprintf(stderr, "[cuda] moe time: %.2fs waiting for the GPU, %.2fs CPU experts, %.2fs cache fills\n",
                         t_gpu_, t_cpu_, t_fill_);
        }
        save_stats();
        if (trace_) std::fclose(trace_);
        if (pred_ready_) {
            if (pred_job_.valid()) pred_job_.get();
            pred_.save();
        }
        for (auto& [k, g] : pre_moe_graphs_) cudaGraphExecDestroy(g);
        for (auto& [k, g] : shared_graphs_) cudaGraphExecDestroy(g);
        for (void* p : dev_bufs_) cudaFree(p);
        for (void* p : host_bufs_) cudaFreeHost(p);
        if (route_done_) cudaEventDestroy(route_done_);
        if (fill_done_) cudaEventDestroy(fill_done_);
        if (stream_) cudaStreamDestroy(stream_);
        if (copy_stream_) cudaStreamDestroy(copy_stream_);
    }

    static int expert_threads(int requested) {
        if (requested > 0) return requested;
        const int hw = (int)std::thread::hardware_concurrency();
        return std::max(1, hw >= 16 ? hw / 2 : hw - 2);
    }

    void warm_experts() {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int> keys;
        for (int k = 0; k < (int)n_keys(); k++)
            if (where_[k].first < 0) keys.push_back(k);
        std::stable_sort(keys.begin(), keys.end(), [&](int a, int b) { return usage_[a] > usage_[b]; });
        std::atomic<size_t> next{0}, bytes{0};
        std::atomic<unsigned> sink{0};
        cpu_.parallel_for(cpu_.size(), [&](int64_t, int64_t) {
            unsigned sum = 0;
            size_t done = 0;
            for (size_t i; (i = next.fetch_add(1)) < keys.size();) {
                const int l = keys[i] / c_.n_experts, e = keys[i] % c_.n_experts;
                for (const gq::GMat* st : {&w_.layers[l].ex_gate, &w_.layers[l].ex_up, &w_.layers[l].ex_down}) {
                    const gq::GMat m = st->slice(e);
                    for (size_t o = 0; o < m.nbytes(); o += 4096) sum += m.data[o];
                    done += m.nbytes();
                }
            }
            sink += sum;
            bytes += done;
        });
        std::fprintf(stderr, "[cuda] touched %.1f GB of experts in RAM in %.1fs\n", bytes / 1073741824.0,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    void reset() override {
        pos_ = 0;
        CUDA_CHECK(cudaMemsetAsync(conv_state_, 0, conv_state_sz_, stream_));
        CUDA_CHECK(cudaMemsetAsync(rec_state_, 0, rec_state_sz_, stream_));
        CUDA_CHECK(cudaMemsetAsync(ple_state_, 0, ple_state_sz_, stream_));
        if (draft_) CUDA_CHECK(cudaMemsetAsync(hlast_, 0, hlast_sz_, stream_));
        if (fill_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, fill_done_, 0));
            fill_pending_ = false;
        }
        for (int i = 0; i < 20000 && cudaStreamQuery(stream_) == cudaErrorNotReady; i++) {}
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    int pos() const override { return pos_; }
    int max_ctx() const override { return max_ctx_; }
    const char* name() const override { return "cuda"; }
    bool cache_stats(CacheStats& s) const override {
        s.hits = hits_;
        s.misses = misses_;
        s.cached = 0;
        for (const auto& pool : slots_)
            for (const Slot& sl : pool) s.cached += sl.key >= 0;
        s.experts = n_keys();
        return true;
    }

    void save_state() override {
        CUDA_CHECK(cudaMemcpyAsync(conv_saved_, conv_state_, conv_state_sz_, cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(rec_saved_, rec_state_, rec_state_sz_, cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(ple_saved_, ple_state_, ple_state_sz_, cudaMemcpyDeviceToDevice, stream_));
        if (draft_) CUDA_CHECK(cudaMemcpyAsync(hlast_saved_, hlast_, hlast_sz_, cudaMemcpyDeviceToDevice, stream_));
        if (fill_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, fill_done_, 0));
            fill_pending_ = false;
        }
        for (int i = 0; i < 20000 && cudaStreamQuery(stream_) == cudaErrorNotReady; i++) {}
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        saved_pos_ = pos_;
    }

    void restore_state() override {
        CUDA_CHECK(cudaMemcpyAsync(conv_state_, conv_saved_, conv_state_sz_, cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(rec_state_, rec_saved_, rec_state_sz_, cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(ple_state_, ple_saved_, ple_state_sz_, cudaMemcpyDeviceToDevice, stream_));
        if (draft_) CUDA_CHECK(cudaMemcpyAsync(hlast_, hlast_saved_, hlast_sz_, cudaMemcpyDeviceToDevice, stream_));
        if (fill_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, fill_done_, 0));
            fill_pending_ = false;
        }
        for (int i = 0; i < 20000 && cudaStreamQuery(stream_) == cudaErrorNotReady; i++) {}
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        pos_ = saved_pos_;
    }

    void forward(const int* tokens, int n, float* logits) override {
        check_tokens(tokens, n);
        for (int done = 0; done < n;) {
            const int m = std::min(NMAX, n - done), p0 = pos_;
            forward_chunk(tokens + done, m, done + m == n ? logits : nullptr, false);
            if (draft_) mtp_run(tokens + done, m, p0, nullptr, true, true);
            done += m;
        }
    }

    int max_draft() const override { return draft_; }

    int draft(int token, int n, int* out) override {
        n = std::min({n, draft_, max_ctx_ - pos_ - 2});
        int cnt = 0;
        mlogits_.resize(c_.vocab);
        for (int k = 0; k < n; k++) {
            mtp_run(&token, 1, pos_ + k, mlogits_.data(), k == 0, false);
            const int best = (int)(std::max_element(mlogits_.begin(), mlogits_.end()) - mlogits_.begin());
            const float mx = mlogits_[best];
            double sum = 0;
            for (float v : mlogits_) sum += std::exp((double)(v - mx));
            if (1.0 / sum < DRAFT_P_MIN) break;
            out[cnt++] = token = best;
        }
        return cnt;
    }

    void forward_all(const int* tokens, int n, float* logits) override {
        if (n < 1 || n > draft_ + 1) throw std::runtime_error("forward_all: bad token count");
        check_tokens(tokens, n);
        spec_pos0_ = pos_;
        spec_toks_.assign(tokens, tokens + n);
        forward_chunk(tokens, n, logits, true);
    }

    void accept(int keep) override {
        const int n = (int)spec_toks_.size();
        if (keep < 1 || keep > n) throw std::runtime_error("accept: bad count");
        if (keep < n) {
            CUDA_CHECK(cudaMemcpyAsync(conv_state_, conv_snap_ + (size_t)(keep - 1) * conv_state_sz_, conv_state_sz_,
                                       cudaMemcpyDeviceToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(rec_state_, rec_snap_ + (size_t)(keep - 1) * rec_state_sz_, rec_state_sz_,
                                       cudaMemcpyDeviceToDevice, stream_));
            if (ple_hist_)
                CUDA_CHECK(cudaMemcpyAsync(ple_state_,
                                           reinterpret_cast<uint8_t*>(ppad_) + (size_t)keep * c_.hc_dim() * 4,
                                           (size_t)ple_hist_ * c_.hc_dim() * 4, cudaMemcpyDeviceToDevice, stream_));
            if (fill_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, fill_done_, 0));
            fill_pending_ = false;
        }
        for (int i = 0; i < 20000 && cudaStreamQuery(stream_) == cudaErrorNotReady; i++) {}
        CUDA_CHECK(cudaStreamSynchronize(stream_));
            pos_ = spec_pos0_ + keep;
        }
        mtp_run(spec_toks_.data(), keep, spec_pos0_, nullptr, true, true);
        spec_toks_.clear();
    }

private:
    void check_tokens(const int* tokens, int n) const {
        if (pos_ + n > max_ctx_) throw std::runtime_error("context overflow");
        for (int t = 0; t < n; t++)
            if (tokens[t] < 0 || tokens[t] >= c_.vocab) throw std::runtime_error("token id out of range");
    }

    uint8_t* dev_alloc(size_t sz, bool may_fail = false) {
        void* p = nullptr;
        cudaError_t e = cudaMalloc(&p, sz);
        if (e != cudaSuccess) {
            if (may_fail) return nullptr;
            throw std::runtime_error("CUDA device allocation failed (" + std::to_string(sz >> 20) + " MB): " +
                                     cudaGetErrorString(e));
        }
        dev_bufs_.push_back(p);
        dev_alloc_bytes_ += sz;
        dev_sizes_[p] = sz;
        return reinterpret_cast<uint8_t*>(p);
    }

    void dev_free(void* p) {
        if (!p) return;
        auto it = dev_sizes_.find(p);
        if (it != dev_sizes_.end()) {
            dev_alloc_bytes_ -= it->second;
            dev_sizes_.erase(it);
        }
        dev_bufs_.erase(std::find(dev_bufs_.begin(), dev_bufs_.end(), p));
        cudaFree(p);
    }

    uint8_t* host_alloc(size_t sz) {
        void* p = nullptr;
        CUDA_CHECK(cudaMallocHost(&p, sz));
        host_bufs_.push_back(p);
        return reinterpret_cast<uint8_t*>(p);
    }

    size_t allocated_device() const { return arena_.allocated() + dev_alloc_bytes_; }

    size_t vram_left() const {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const size_t own = allocated_device();
        if (sys_free0_) free_b = std::min(free_b, sys_free0_ > own ? sys_free0_ - own : 0);
        return free_b > VRAM_MARGIN ? free_b - VRAM_MARGIN : 0;
    }

    CM up_dense(const gq::GMat& m, bool force_f32 = false) {
        CM g;
        g.rows = m.rows;
        g.cols = m.cols;
        if (m.cols % 32) throw std::runtime_error("matrix width not a multiple of 32");
        g.q8 = m.type == GType::Q8_0 && !force_f32;
        const size_t nb = (size_t)m.cols / 32, row_bytes = g.q8 ? (size_t)m.cols : (size_t)m.cols * 4;
        g.w = arena_.alloc((size_t)m.rows * row_bytes);
        if (g.q8) g.s = arena_.alloc((size_t)m.rows * nb * 4);
        const int batch = std::max<int>(1, (int)(((size_t)48 << 20) / row_bytes));
        std::vector<uint8_t> wbuf;
        std::vector<float> sbuf;
        for (int r0 = 0; r0 < m.rows; r0 += batch) {
            const int nr = std::min(batch, m.rows - r0);
            wbuf.resize((size_t)nr * row_bytes);
            sbuf.resize((size_t)nr * nb);
            cpu_.parallel_for(nr, [&](int64_t b, int64_t e) {
                for (int64_t r = b; r < e; r++) {
                    if (g.q8) unpack_q8_row(m.row(r0 + r), m.cols, &wbuf[(size_t)r * row_bytes], &sbuf[(size_t)r * nb]);
                    else gq::dequant_row(m.type, m.row(r0 + r), m.cols, (float*)&wbuf[(size_t)r * row_bytes]);
                }
            }, 16);
            CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(g.w + (size_t)r0 * row_bytes), wbuf.data(), wbuf.size(),
                                  cudaMemcpyHostToDevice));
            if (g.q8)
                CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(g.s + (size_t)r0 * nb * 4), sbuf.data(), sbuf.size() * 4,
                                      cudaMemcpyHostToDevice));
        }
        return g;
    }

    uint64_t up_f32(const std::vector<float>& v) {
        uint64_t p = arena_.alloc(v.size() * 4);
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(p), v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        return p;
    }

    CM up_row(const std::vector<float>& v) {
        CM g;
        g.rows = 1;
        g.cols = (int)v.size();
        g.w = up_f32(v);
        return g;
    }

    CHc up_hc(const Q4Hc& h) {
        CHc g;
        g.norm = up_f32(h.norm);
        g.down = up_dense(h.down);
        g.up = up_dense(h.up);
        g.has_inject = h.inject.data != nullptr;
        if (g.has_inject) g.inject = up_dense(h.inject);
        return g;
    }

    uint64_t up_norm_minus1(std::vector<float> v) {
        for (float& x : v) x -= 1.0f;
        return up_f32(v);
    }

    void upload_weights() {
        auto t0 = std::chrono::steady_clock::now();
        output_ = up_dense(w_.output);
        hc_head_ = up_hc(w_.hc_head);
        int nf = 0, nl = 0, nq = 0;
        const int n_blocks = c_.n_layers + (draft_ ? 1 : 0);
        if (draft_) {
            mtp_eh_ = up_dense(w_.mtp_eh);
            mtp_enorm_ = up_norm_minus1(w_.mtp_enorm);
            mtp_hnorm_ = up_f32(w_.mtp_hnorm);
            mtp_head_ = up_hc(w_.mtp_head);
        }
        for (int l = 0; l < n_blocks; l++) {
            const Q4Layer& W = w_.layers[l];
            CLayer g;
            g.hc_attn = up_hc(W.hc_attn);
            g.hc_ffn = up_hc(W.hc_ffn);
            if (W.full_attn) {
                slot_.push_back(nf++);
                g.q = up_dense(W.q); g.k = up_dense(W.k); g.v = up_dense(W.v); g.o = up_dense(W.o);
                g.qn = up_norm_minus1(W.q_norm);
                g.kn = up_norm_minus1(W.k_norm);
            } else {
                slot_.push_back(nl++);
                g.qkv = up_dense(W.qkv); g.z = up_dense(W.z); g.out = up_dense(W.out);
                g.alpha = up_dense(W.alpha); g.beta = up_dense(W.beta);
                g.conv = up_f32(W.conv1d);
                g.neg_a = up_f32(W.ssm_a);
                g.dt_bias = up_f32(W.dt_bias);
                g.lnorm = up_f32(W.ssm_norm);
            }
            qsa_slot_.push_back(W.full_attn && W.compress > 0 ? nq++ : -1);
            if (W.ple) {
                g.ple_key = up_dense(W.ple_key); g.ple_value = up_dense(W.ple_value);
                g.ple_wk = up_f32(W.ple_norm_key); g.ple_wq = up_f32(W.ple_norm_query); g.ple_wc = up_f32(W.ple_norm_conv);
                g.ple_conv = up_f32(W.ple_conv1d);
            }
            g.router = up_dense(W.router);
            g.sgate = up_row(W.shared_gate);
            const bool sq8 = W.sh_gate.type == GType::Q8_0 && W.sh_up.type == GType::Q8_0;
            g.sh_gate = up_dense(W.sh_gate, !sq8);
            g.sh_up = up_dense(W.sh_up, !sq8);
            g.sh_down = up_dense(W.sh_down);
            L_.push_back(g);
        }
        const int half = c_.rope_dim / 2;
        std::vector<float> rope((size_t)max_ctx_ * c_.rope_dim);
        for (int p = 0; p < max_ctx_; p++)
            for (int i = 0; i < half; i++) {
                float inv = (float)(1.0 / std::pow((double)c_.rope_theta, (2.0 * i) / c_.rope_dim));
                float ang = (float)p * inv;
                rope[(size_t)p * c_.rope_dim + 2 * i] = std::cos(ang);
                rope[(size_t)p * c_.rope_dim + 2 * i + 1] = std::sin(ang);
            }
        rope_ = up_f32(rope);
        std::fprintf(stderr, "[cuda] dense weights uploaded in %.1fs\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    void alloc_state() {
        const int H = c_.hidden, HC = c_.hc_dim(), TK = c_.top_k;
        const int n_lin = (int)std::count(c_.is_full_attn.begin(), c_.is_full_attn.end(), false);
        const int n_full = (int)L_.size() - n_lin;
        const int n_qsa = (int)std::count_if(qsa_slot_.begin(), qsa_slot_.end(), [](int s) { return s >= 0; });
        ple_hist_ = c_.ple_ngram > 0 ? (c_.ple_kernel - 1) * c_.ple_ngram : 0;
        int max_blocks = 1;
        for (size_t l = 0; l < L_.size(); l++)
            if (w_.layers[l].compress > 0) max_blocks = std::max(max_blocks, max_ctx_ / w_.layers[l].compress + 1);
        mask_words_ = (max_blocks + 31) / 32;
        toks_.assign(max_ctx_, 0);

        auto act = [&](size_t floats) { return arena_.alloc(floats * 4); };
        x_ = act((size_t)NMAX * HC);
        xnorm_ = act((size_t)NMAX * HC);
        gate_ = act((size_t)NMAX * HC);
        lo_ = act((size_t)NMAX * c_.hc_rank);
        inj_ = act((size_t)NMAX * c_.hc);
        xn_ = act((size_t)NMAX * H);
        mix_ = act((size_t)NMAX * H);
        qkv_ = act((size_t)NMAX * c_.conv_dim());
        z_ = act((size_t)NMAX * c_.lin_value_dim());
        a_ = act((size_t)NMAX * c_.lin_v_heads);
        b_ = act((size_t)NMAX * c_.lin_v_heads);
        core_ = act((size_t)NMAX * c_.lin_value_dim());
        qg_ = act((size_t)NMAX * c_.n_heads * c_.head_dim * 2);
        kb_ = act((size_t)NMAX * c_.n_kv_heads * c_.head_dim);
        vb_ = act((size_t)NMAX * c_.n_kv_heads * c_.head_dim);
        qo_ = act((size_t)NMAX * c_.n_heads * c_.head_dim);
        att_ = act((size_t)NMAX * c_.n_heads * c_.head_dim);
        part_ = act((size_t)ATT_PART_MAX * c_.n_heads * (c_.head_dim + 2));
        sact_ = act((size_t)NMAX * c_.shared_inter);
        sy_ = act((size_t)NMAX * H);
        sgate_ = act(NMAX);
        act_ = act((size_t)NMAX * TK * c_.moe_inter);
        ey_ = act((size_t)NMAX * TK * H);
        cy16_ = arena_.alloc((size_t)NMAX * TK * H * 2);
        rlogits_ = act((size_t)NMAX * c_.n_experts);
        selw_ = act((size_t)NMAX * TK);
        comb_ = arena_.alloc((size_t)NMAX * TK * 4);
        sel_ = arena_.alloc((size_t)NMAX * TK * 4);
        logits_ = arena_.alloc((size_t)c_.vocab * 4 * (draft_ + 1));
        dmask_ = arena_.alloc((size_t)NMAX * mask_words_ * 4);
        if (draft_) {
            mx_ = act((size_t)NMAX * HC);
            mh_ = act((size_t)NMAX * HC);
            mcat_ = act((size_t)NMAX * HC * 2);
            men_ = act((size_t)NMAX * H);
            memb_ = act((size_t)NMAX * H);
        }
        if (ple_hist_) {
            emb_ = act((size_t)NMAX * H);
            pkey_ = act((size_t)NMAX * HC);
            pval_ = act((size_t)NMAX * H);
            pgated_ = act((size_t)NMAX * HC);
            ppad_ = act((size_t)(NMAX + ple_hist_) * HC);
        }

        const size_t max_qcols = (size_t)std::max({HC, 2 * H, c_.lin_value_dim(), c_.n_heads * c_.head_dim,
                                                   c_.shared_inter, c_.moe_inter});
        xq_ = arena_.alloc((size_t)NT * max_qcols);
        xs_ = arena_.alloc((size_t)NT * (max_qcols / 32) * 4);
        d_attn_params_ = arena_.alloc(sizeof(AttnDynParams));
        h_attn_params_ = reinterpret_cast<AttnDynParams*>(host_alloc(sizeof(AttnDynParams)));

        const size_t ne = (size_t)NMAX * TK;
        htab_toks_ = align256(ne * sizeof(ExpertEntry));
        htab_comb_ = align256(htab_toks_ + ne * 4);
        dtab_ = arena_.alloc(htab_comb_ + ne * 8);

        n_kv_slots_ = n_full;
        const char* ks = std::getenv("QW_KV_START");
        kv_cap_ = std::min(max_ctx_, ks ? std::max(1, std::atoi(ks)) : KV_START);
        const size_t kv = (size_t)n_full * kv_cap_ * c_.n_kv_heads * c_.head_dim * 2;
        kv_bytes_ = 2 * kv;
        kcache_ = dev_alloc(std::max<size_t>(kv, 256));
        vcache_ = dev_alloc(std::max<size_t>(kv, 256));

        conv_state_sz_ = std::max<size_t>((size_t)n_lin * (c_.conv_kernel - 1) * c_.conv_dim() * 4, 256);
        rec_state_sz_ = std::max<size_t>((size_t)n_lin * c_.lin_v_heads * c_.lin_k_dim * c_.lin_v_dim * 4, 256);
        ple_state_sz_ = std::max<size_t>((size_t)ple_hist_ * HC * 4, 256);
        conv_state_ = dev_alloc(conv_state_sz_);
        rec_state_ = dev_alloc(rec_state_sz_);
        ple_state_ = dev_alloc(ple_state_sz_);
        conv_saved_ = dev_alloc(conv_state_sz_);
        rec_saved_ = dev_alloc(rec_state_sz_);
        ple_saved_ = dev_alloc(ple_state_sz_);
        if (draft_) {
            hme_ = host_alloc((size_t)NMAX * H * 4);
            hlast_sz_ = (size_t)HC * 4;
            hlast_ = dev_alloc(hlast_sz_);
            hlast_saved_ = dev_alloc(hlast_sz_);
            conv_snap_ = dev_alloc(conv_state_sz_ * draft_);
            rec_snap_ = dev_alloc(rec_state_sz_ * draft_);
        }
        hx_ = host_alloc((size_t)NMAX * HC * 4);
        hemb_ = host_alloc((size_t)NMAX * H * 4);
        hsel_ = host_alloc((size_t)NMAX * TK * 4);
        hxn_ = host_alloc((size_t)NMAX * H * 4);
        hcy_ = host_alloc((size_t)NMAX * TK * H * 2);
        htab_ = host_alloc(htab_comb_ + ne * 4);
        hlog_ = host_alloc((size_t)c_.vocab * 4 * (draft_ + 1));
        hqsa_ = host_alloc(std::max<size_t>((size_t)n_qsa * NMAX * H * 4, 256));
        hmask_ = host_alloc((size_t)NMAX * mask_words_ * 4);
        staging_ = host_alloc(STAGING);
    }

    size_t n_keys() const { return L_.size() * c_.n_experts; }
    uint64_t slot_addr(std::pair<int, int> cs) const {
        return reinterpret_cast<uint64_t>(slots_[cs.first][cs.second].buf) + slots_[cs.first][cs.second].off;
    }

    static bool to_codec(GType t, Codec& out) {
        switch (t) {
            case GType::Q4_K: out = Codec::Q4K; return true;
            case GType::Q5_K: out = Codec::Q5K; return true;
            case GType::Q8_0: out = Codec::Q8; return true;
            case GType::Q5_1: out = Codec::Q51; return true;
            default: return false;
        }
    }

    static void write_part(const gq::GMat& m, uint8_t* d, uint32_t s_off_rel) {
        if (m.type != GType::Q8_0) {
            std::memcpy(d, m.data, m.nbytes());
            return;
        }
        float* s = reinterpret_cast<float*>(d + s_off_rel);
        for (int r = 0; r < m.rows; r++)
            unpack_q8_row(m.row(r), m.cols, d + (size_t)r * m.cols, s + (size_t)r * (m.cols / 32));
    }

    void write_expert(int key, uint8_t* d) const {
        const int l = key / c_.n_experts, e = key % c_.n_experts;
        const Q4Layer& W = w_.layers[l];
        const CLayer& g = L_[l];
        write_part(W.ex_gate.slice(e), d, g.s_gate);
        write_part(W.ex_up.slice(e), d + g.off_up, g.s_up - g.off_up);
        write_part(W.ex_down.slice(e), d + g.off_down, g.s_down - g.off_down);
    }

    bool release_expert_chunk() {
        size_t k = 0;
        for (size_t i = 1; i < slots_.size(); i++)
            if (slots_[i].size() > slots_[k].size()) k = i;
        if (slots_.empty() || slots_[k].empty()) return false;
        uint8_t* b = slots_[k].back().buf;
        while (!slots_[k].empty() && slots_[k].back().buf == b) {
            if (slots_[k].back().key >= 0) where_[slots_[k].back().key] = {-1, -1};
            slots_[k].pop_back();
        }
        dev_free(b);
        return true;
    }

    void ensure_kv(int tokens) {
        if (tokens <= kv_cap_) return;
        const int cap = (int)std::min<int64_t>(max_ctx_, std::max<int64_t>(2ll * kv_cap_, tokens));
        const size_t per = (size_t)c_.n_kv_heads * c_.head_dim * 2;
        const size_t size = (size_t)n_kv_slots_ * cap * per;
        uint8_t* nb[2] = {nullptr, nullptr};
        for (uint8_t*& b : nb) {
            while (vram_left() < size && release_expert_chunk()) {}
            while (!(b = dev_alloc(size, true)))
                if (!release_expert_chunk()) throw std::runtime_error("out of VRAM growing the K/V caches");
        }
        for (int s = 0; s < n_kv_slots_; s++) {
            CUDA_CHECK(cudaMemcpyAsync(nb[0] + (size_t)s * cap * per, kcache_ + (size_t)s * kv_cap_ * per,
                                       (size_t)kv_cap_ * per, cudaMemcpyDeviceToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(nb[1] + (size_t)s * cap * per, vcache_ + (size_t)s * kv_cap_ * per,
                                       (size_t)kv_cap_ * per, cudaMemcpyDeviceToDevice, stream_));
        }
        if (fill_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, fill_done_, 0));
            fill_pending_ = false;
        }
        for (int i = 0; i < 20000 && cudaStreamQuery(stream_) == cudaErrorNotReady; i++) {}
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        dev_free(kcache_);
        dev_free(vcache_);
        kcache_ = nb[0];
        vcache_ = nb[1];
        kv_cap_ = cap;
        // Invalidate attention CUDA graphs because kcache_/vcache_ pointers changed
        for (auto it = pre_moe_graphs_.begin(); it != pre_moe_graphs_.end();) {
            if ((it->first >> 56) == 1ull) {
                cudaGraphExecDestroy(it->second);
                it = pre_moe_graphs_.erase(it);
            } else {
                ++it;
            }
        }
        size_t n_slots = 0;
        for (auto& s : slots_) n_slots += s.size();
        std::fprintf(stderr, "[cuda] K/V caches grown to %d tokens (%.1f GB); %zu expert slots cached\n", cap,
                     2 * size / 1073741824.0, n_slots);
    }

    void setup_experts() {
        for (int l = 0; l < (int)L_.size(); l++) {
            const Q4Layer& W = w_.layers[l];
            CLayer& g = L_[l];
            if (!to_codec(W.ex_gate.type, g.cu) || !to_codec(W.ex_down.type, g.cd) ||
                W.ex_gate.type != W.ex_up.type || W.ex_gate.type == GType::Q5_1)
                throw std::runtime_error(std::string("layer ") + std::to_string(l) + ": unsupported expert quantization");
            auto planar_w = [](const gq::GMat& m) { return (uint32_t)align256((size_t)m.rows * m.cols); };
            const gq::GMat ga = W.ex_gate.slice(0), ua = W.ex_up.slice(0), da = W.ex_down.slice(0);
            size_t o = 0;
            g.s_gate = (uint32_t)o + (ga.type == GType::Q8_0 ? planar_w(ga) : 0);
            o += expert_part_bytes(ga);
            g.off_up = (uint32_t)o;
            g.s_up = (uint32_t)o + (ua.type == GType::Q8_0 ? planar_w(ua) : 0);
            o += expert_part_bytes(ua);
            g.off_down = (uint32_t)o;
            g.s_down = (uint32_t)o + (da.type == GType::Q8_0 ? planar_w(da) : 0);
            o += expert_part_bytes(da);
            g.slot_size = o;
            auto it = std::find(cls_size_.begin(), cls_size_.end(), o);
            g.cls = (int)(it - cls_size_.begin());
            if (it == cls_size_.end()) cls_size_.push_back(o);
        }
        usage_.assign(n_keys(), 0);
        recent_.assign(n_keys(), 0);
        if (const char* ev = std::getenv("QW_EMA")) ema_ = (float)std::atof(ev);
        if (const char* ev = std::getenv("QW_EMA_MARGIN")) ema_margin_ = (float)std::atof(ev);
        heat_.assign(n_keys(), 0.f);
        use_pred_ = std::getenv("QW_PRED") != nullptr;
        if (const char* ev = std::getenv("QW_PRED_MARGIN")) pred_margin_ = (float)std::atof(ev);
        load_stats();

        const size_t own = allocated_device(), left = vram_left();
        size_t budget = opt_.expert_cache_set ? (size_t)opt_.expert_cache_mb << 20 : left;
        budget = std::min(budget, left);
        std::fprintf(stderr, "[cuda] VRAM: %.1f GB free at start, %.1f GB dense weights and state, %.1f GB K/V caches "
                             "for the first %d tokens of context, %.1f GB left for experts\n",
                     sys_free0_ / 1073741824.0, (own - kv_bytes_) / 1073741824.0, kv_bytes_ / 1073741824.0, kv_cap_,
                     left / 1073741824.0);

        const size_t nc = cls_size_.size();
        std::vector<size_t> cls_keys(nc, 0);
        size_t total = 0;
        for (int l = 0; l < (int)L_.size(); l++) {
            cls_keys[L_[l].cls] += c_.n_experts;
            total += L_[l].slot_size * c_.n_experts;
        }
        slots_.resize(nc);
        size_t n_slots = 0, slot_bytes = 0;
        for (size_t k = 0; k < nc; k++) {
            const double share = (double)(cls_size_[k] * cls_keys[k]) / (double)total;
            const size_t want = std::min<size_t>(cls_keys[k], (size_t)(budget * share / cls_size_[k]));
            const size_t per_chunk = CHUNK / cls_size_[k];
            while (slots_[k].size() < want) {
                const size_t cnt = std::min(per_chunk, want - slots_[k].size());
                uint8_t* b = dev_alloc(cnt * cls_size_[k], true);
                if (!b) break;
                for (size_t i = 0; i < cnt; i++) slots_[k].push_back({b, i * cls_size_[k], -1, 0});
            }
            n_slots += slots_[k].size();
            slot_bytes += slots_[k].size() * cls_size_[k];
        }
        where_.assign(n_keys(), {-1, -1});
        std::fprintf(stderr, "[cuda] expert cache: %zu slots = %.1f GB of VRAM (%.0f%% of %zu experts, %.1f GB); "
                             "%d CPU threads compute the rest from RAM\n",
                     n_slots, slot_bytes / 1073741824.0, 100.0 * n_slots / n_keys(), n_keys(), total / 1073741824.0,
                     cpu_.size());
        preload();
    }

    void preload() {
        std::vector<int> order(n_keys());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            const int ea = a % c_.n_experts, eb = b % c_.n_experts;
            return usage_[a] != usage_[b] ? usage_[a] > usage_[b] : ea < eb;
        });
        auto t0 = std::chrono::steady_clock::now();
        std::vector<size_t> used(slots_.size(), 0);
        std::vector<std::pair<int, int>> todo;
        for (int key : order) {
            const int k = L_[key / c_.n_experts].cls;
            if (used[k] == slots_[k].size()) continue;
            const int sidx = (int)used[k]++;
            slots_[k][sidx].key = key;
            where_[key] = {k, sidx};
            todo.push_back({key, sidx});
        }
        const size_t batch = std::max<size_t>(1, (STAGING / 2) / cls_size_[0] / 2);
        std::vector<std::vector<uint8_t>> tmp(batch);
        for (size_t i = 0; i < todo.size(); i += batch) {
            const size_t cnt = std::min(batch, todo.size() - i);
            cpu_.parallel_for((int64_t)cnt, [&](int64_t b, int64_t e) {
                for (int64_t j = b; j < e; j++) {
                    tmp[j].resize(L_[todo[i + j].first / c_.n_experts].slot_size);
                    write_expert(todo[i + j].first, tmp[j].data());
                }
            });
            for (size_t j = 0; j < cnt; j++) {
                const Slot& sl = slots_[where_[todo[i + j].first].first][todo[i + j].second];
                CUDA_CHECK(cudaMemcpyAsync(sl.buf + sl.off, tmp[j].data(), tmp[j].size(), cudaMemcpyHostToDevice, copy_stream_));
            }
            CUDA_CHECK(cudaStreamSynchronize(copy_stream_));
        }
        std::fprintf(stderr, "[cuda] preloaded %zu experts (%s) in %.1fs\n", todo.size(),
                     std::any_of(usage_.begin(), usage_.end(), [](uint64_t u) { return u > 0; }) ? "by usage stats" : "uniform",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }

    struct FillJob {
        int key, cls, slot;
        size_t staging_off, bytes;
    };

    void plan_fill(int key, size_t& staging_used, std::vector<FillJob>& jobs) {
        const int k = L_[key / c_.n_experts].cls;
        auto& sl = slots_[k];
        int v = -1;
        for (int i = 0; i < (int)sl.size(); i++) {
            if (sl[i].key < 0) { v = i; break; }
            if (sl[i].tick == tick_) continue;
            if (v < 0 || heat(sl[i].key) < heat(sl[v].key)) v = i;
        }
        if (v < 0) return;
        if (sl[v].key >= 0) {
            if (heat(key) < heat(sl[v].key) + fill_margin()) return;
            where_[sl[v].key] = {-1, -1};
        }
        if (staging_used + cls_size_[k] > STAGING) return;  // cap to single staging buffer (256 MB >= 64 experts)
        jobs.push_back({key, k, v, staging_used, cls_size_[k]});
        staging_used += cls_size_[k];
        sl[v].key = key;
        sl[v].tick = tick_;
        where_[key] = {k, v};
        fills_++;
    }

    void load_stats() {
        if (opt_.stats_path.empty()) return;
        std::ifstream f(opt_.stats_path, std::ios::binary);
        if (!f) return;
        std::vector<uint64_t> v(n_keys(), 0);
        f.read((char*)v.data(), v.size() * 8);
        if ((size_t)f.gcount() >= (size_t)c_.n_layers * c_.n_experts * 8) usage_ = v;
    }
    void save_stats() {
        if (opt_.stats_path.empty()) return;
        std::ofstream f(opt_.stats_path, std::ios::binary);
        f.write((const char*)usage_.data(), usage_.size() * 8);
    }

    // ---- kernel launch wrappers ---------------------------------------------------------------
    void ensure_qx(uint64_t x, int n, int cols, int x_stride) {
        if (!use_graph_ && qx_valid_ && qx_x_ == x && qx_n_ == n && qx_cols_ == cols && qx_stride_ == x_stride) return;
        quantize_x_q8_kernel<<<dim3(cols >> 5, n), 32, 0, stream_>>>(
            reinterpret_cast<const float*>(x), reinterpret_cast<int8_t*>(xq_), reinterpret_cast<float*>(xs_),
            cols, x_stride);
        qx_valid_ = true;
        qx_x_ = x; qx_n_ = n; qx_cols_ = cols; qx_stride_ = x_stride;
    }

    void mat(const CM& m, uint64_t x, uint64_t y, int n, int x_stride, uint32_t post = 0, float post_scale = 0.0f) {
        if (y == qx_x_) qx_valid_ = false;
        if (n >= GEMM_MIN && m.rows % 64 == 0) {
            dim3 grid(groups(m.rows, 64), groups(n, 64));
            if (m.q8)
                q4_gemm_wmma_kernel<Codec::Q8, false, false><<<grid, 256, 0, stream_>>>(
                    m.w, m.s, 0, 0, x, y, 0, 0, 0, 0, m.rows, m.cols, n, x_stride, m.rows, post, post_scale);
            else
                q4_gemm_wmma_kernel<Codec::F32, false, false><<<grid, 256, 0, stream_>>>(
                    m.w, 0, 0, 0, x, y, 0, 0, 0, 0, m.rows, m.cols, n, x_stride, m.rows, post, post_scale);
        } else if (use_dp4a_ && m.q8 && n <= NT) {
            ensure_qx(x, n, m.cols, x_stride);
            if (m.rows <= 256 && m.cols >= 2048) {
                q4_matmul_wide_q8_dp4a_kernel<<<m.rows, 256, 0, stream_>>>(
                    m.w, m.s, reinterpret_cast<const int8_t*>(xq_), reinterpret_cast<const float*>(xs_), y,
                    m.rows, m.cols, n, m.rows, post, post_scale);
            } else {
                dim3 grid(groups(m.rows, 8), 1);
                q4_matmul_q8_dp4a_kernel<false, false><<<grid, 256, 0, stream_>>>(
                    m.w, m.s, 0, 0, reinterpret_cast<const int8_t*>(xq_), reinterpret_cast<const float*>(xs_), y, 0, 0,
                    0, 0, m.rows, m.cols, n, m.rows, post, post_scale);
            }
        } else if (m.rows <= 256 && m.cols >= 2048) {
            dim3 grid(m.rows, groups(n, NT));
            if (m.q8)
                q4_matmul_wide_kernel<Codec::Q8><<<grid, 256, 0, stream_>>>(
                    m.w, m.s, x, y, m.rows, m.cols, n, x_stride, m.rows, post, post_scale);
            else
                q4_matmul_wide_kernel<Codec::F32><<<grid, 256, 0, stream_>>>(
                    m.w, 0, x, y, m.rows, m.cols, n, x_stride, m.rows, post, post_scale);
        } else {
            dim3 grid(groups(m.rows, 8), groups(n, NT));
            if (m.q8)
                q4_matmul_kernel<Codec::Q8, false, false><<<grid, 256, 0, stream_>>>(
                    m.w, m.s, 0, 0, x, y, 0, 0, 0, 0, m.rows, m.cols, n, x_stride, m.rows, post, post_scale);
            else
                q4_matmul_kernel<Codec::F32, false, false><<<grid, 256, 0, stream_>>>(
                    m.w, 0, 0, 0, x, y, 0, 0, 0, 0, m.rows, m.cols, n, x_stride, m.rows, post, post_scale);
        }
    }

    void mat_swiglu(const CM& g, const CM& u, uint64_t x, uint64_t y, int n) {
        if (y == qx_x_) qx_valid_ = false;
        if (n >= GEMM_MIN && g.rows % 64 == 0) {
            dim3 grid(groups(g.rows, 64), groups(n, 64));
            if (g.q8)
                q4_gemm_wmma_kernel<Codec::Q8, false, true><<<grid, 256, 0, stream_>>>(
                    g.w, g.s, u.w, u.s, x, y, 0, 0, 0, 0, g.rows, g.cols, n, g.cols, g.rows, 0, 0.0f);
            else
                q4_gemm_wmma_kernel<Codec::F32, false, true><<<grid, 256, 0, stream_>>>(
                    g.w, 0, u.w, 0, x, y, 0, 0, 0, 0, g.rows, g.cols, n, g.cols, g.rows, 0, 0.0f);
        } else if (use_dp4a_ && g.q8 && n <= NT) {
            ensure_qx(x, n, g.cols, g.cols);
            dim3 grid(groups(g.rows, 8), 1);
            q4_matmul_q8_dp4a_kernel<false, true><<<grid, 256, 0, stream_>>>(
                g.w, g.s, u.w, u.s, reinterpret_cast<const int8_t*>(xq_), reinterpret_cast<const float*>(xs_), y, 0, 0,
                0, 0, g.rows, g.cols, n, g.rows, 0, 0.0f);
        } else {
            dim3 grid(groups(g.rows, 8), groups(n, NT));
            if (g.q8)
                q4_matmul_kernel<Codec::Q8, false, true><<<grid, 256, 0, stream_>>>(
                    g.w, g.s, u.w, u.s, x, y, 0, 0, 0, 0, g.rows, g.cols, n, g.cols, g.rows, 0, 0.0f);
            else
                q4_matmul_kernel<Codec::F32, false, true><<<grid, 256, 0, stream_>>>(
                    g.w, 0, u.w, 0, x, y, 0, 0, 0, 0, g.rows, g.cols, n, g.cols, g.rows, 0, 0.0f);
        }
    }

    template <bool SwiGLU>
    void dispatch_expert(Codec codec, bool gemm, uint64_t tab, uint64_t tk, uint64_t x, uint64_t y, uint32_t woff1,
                         uint32_t soff1, uint32_t woff2, uint32_t soff2, uint32_t rows, uint32_t cols, uint32_t x_stride,
                         uint32_t y_stride, uint32_t x_mode, uint32_t ne) {
        const uint32_t rpw = gemm ? 64u : 8u;
        dim3 grid(groups(rows, rpw), ne);
        auto launch = [&](auto tag) {
            constexpr Codec C = decltype(tag)::value;
            if (gemm)
                q4_gemm_wmma_kernel<C, true, SwiGLU><<<grid, 256, 0, stream_>>>(
                    tab, tk, 0, 0, x, y, woff1, soff1, woff2, soff2, rows, cols, x_mode, x_stride, y_stride, 0, 0.0f);
            else
                q4_matmul_kernel<C, true, SwiGLU><<<grid, 256, 0, stream_>>>(
                    tab, tk, 0, 0, x, y, woff1, soff1, woff2, soff2, rows, cols, x_mode, x_stride, y_stride, 0, 0.0f);
        };
        switch (codec) {
            case Codec::Q4K: launch(std::integral_constant<Codec, Codec::Q4K>{}); break;
            case Codec::Q5K: launch(std::integral_constant<Codec, Codec::Q5K>{}); break;
            case Codec::Q8: launch(std::integral_constant<Codec, Codec::Q8>{}); break;
            case Codec::Q51: launch(std::integral_constant<Codec, Codec::Q51>{}); break;
            default: throw std::runtime_error("unsupported expert codec");
        }
    }

    void hc_mix(const CHc& h, uint64_t x, int n) {
        qx_valid_ = false;
        const int H = c_.hidden, HC = c_.hc_dim();
        q4_hc_norm_kernel<<<dim3(n, c_.hc), 256, 0, stream_>>>(
            reinterpret_cast<const float*>(x), reinterpret_cast<const float*>(h.norm),
            reinterpret_cast<float*>(xnorm_), H, c_.hc, c_.rms_eps);
        mat(h.down, xnorm_, lo_, n, HC, 1, 1.0f / c_.hc);
        if (h.has_inject) mat(h.inject, xnorm_, inj_, n, HC);
        qx_valid_ = false;
        mat(h.up, lo_, gate_, n, c_.hc_rank);
        q4_hc_mix_kernel<<<dim3(groups(H, 256), n), 256, 0, stream_>>>(
            reinterpret_cast<const float*>(xnorm_), reinterpret_cast<const float*>(gate_),
            reinterpret_cast<float*>(xn_), H, c_.hc);
        qx_valid_ = false;
    }

    void hc_combine(uint64_t x, int n) {
        q4_hc_combine_kernel<<<dim3(groups(c_.hidden, 256), n), 256, 0, stream_>>>(
            reinterpret_cast<float*>(x), reinterpret_cast<const float*>(inj_),
            reinterpret_cast<const float*>(mix_), c_.hidden, c_.hc);
    }

    void ple(int l, int n) {
        qx_valid_ = false;
        const CLayer& g = L_[l];
        const int H = c_.hidden, HC = c_.hc_dim();
        const size_t hb = (size_t)ple_hist_ * HC * 4;
        mat(g.ple_key, emb_, pkey_, n, H);
        mat(g.ple_value, emb_, pval_, n, H);
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(ppad_), ple_state_, hb, cudaMemcpyDeviceToDevice, stream_));
        q4_ple_gate_kernel<<<dim3(n, c_.hc), 256, 0, stream_>>>(
            reinterpret_cast<const float*>(pkey_), reinterpret_cast<const float*>(pval_),
            reinterpret_cast<const float*>(x_), reinterpret_cast<const float*>(g.ple_wk),
            reinterpret_cast<const float*>(g.ple_wq), reinterpret_cast<const float*>(g.ple_wc),
            reinterpret_cast<float*>(pgated_), reinterpret_cast<float*>(ppad_), H, c_.hc, ple_hist_, c_.rms_eps);
        q4_ple_conv_kernel<<<dim3(groups(HC, 256), n), 256, 0, stream_>>>(
            reinterpret_cast<float*>(x_), reinterpret_cast<const float*>(pgated_),
            reinterpret_cast<const float*>(ppad_), reinterpret_cast<const float*>(g.ple_conv),
            HC, c_.ple_kernel, c_.ple_ngram, ple_hist_);
        CUDA_CHECK(cudaMemcpyAsync(ple_state_, reinterpret_cast<uint8_t*>(ppad_) + (size_t)n * HC * 4, hb,
                                   cudaMemcpyDeviceToDevice, stream_));
    }

    void deltanet(int l, int n, bool snap) {
        const CLayer& g = L_[l];
        const int s = slot_[l], H = c_.hidden;
        mat(g.qkv, xn_, qkv_, n, H);
        mat(g.z, xn_, z_, n, H);
        mat(g.alpha, xn_, a_, n, H);
        mat(g.beta, xn_, b_, n, H);
        const size_t cs = (size_t)(c_.conv_kernel - 1) * c_.conv_dim() * 4;
        const uint64_t cs_snap = snap ? reinterpret_cast<uint64_t>(conv_snap_ + s * cs) : 0;
        if (snap)
            conv1d_kernel<true><<<groups(c_.conv_dim(), 256), 256, 0, stream_>>>(
                reinterpret_cast<float*>(qkv_), reinterpret_cast<const float*>(g.conv),
                reinterpret_cast<float*>(conv_state_ + s * cs), n, c_.conv_dim(), cs_snap, (uint32_t)conv_state_sz_);
        else
            conv1d_kernel<false><<<groups(c_.conv_dim(), 256), 256, 0, stream_>>>(
                reinterpret_cast<float*>(qkv_), reinterpret_cast<const float*>(g.conv),
                reinterpret_cast<float*>(conv_state_ + s * cs), n, c_.conv_dim(), 0, 0);

        const size_t rs = (size_t)c_.lin_v_heads * c_.lin_k_dim * c_.lin_v_dim * 4;
        const uint64_t rs_snap = snap ? reinterpret_cast<uint64_t>(rec_snap_ + s * rs) : 0;
        if (snap)
            deltanet_tiled_kernel<true><<<dim3(c_.lin_v_heads, c_.lin_v_dim / 16), 128, 0, stream_>>>(
                reinterpret_cast<const float*>(qkv_), reinterpret_cast<const float*>(a_),
                reinterpret_cast<const float*>(b_), reinterpret_cast<const float*>(g.neg_a),
                reinterpret_cast<const float*>(g.dt_bias), reinterpret_cast<float*>(rec_state_ + s * rs),
                reinterpret_cast<float*>(core_), n, c_.conv_dim(), c_.lin_key_dim(), c_.lin_v_heads,
                c_.lin_v_heads / c_.lin_k_heads, rs_snap, (uint32_t)rec_state_sz_);
        else
            deltanet_tiled_kernel<false><<<dim3(c_.lin_v_heads, c_.lin_v_dim / 16), 128, 0, stream_>>>(
                reinterpret_cast<const float*>(qkv_), reinterpret_cast<const float*>(a_),
                reinterpret_cast<const float*>(b_), reinterpret_cast<const float*>(g.neg_a),
                reinterpret_cast<const float*>(g.dt_bias), reinterpret_cast<float*>(rec_state_ + s * rs),
                reinterpret_cast<float*>(core_), n, c_.conv_dim(), c_.lin_key_dim(), c_.lin_v_heads,
                c_.lin_v_heads / c_.lin_k_heads, 0, 0);

        gated_norm_sigmoid_kernel<<<dim3(n, c_.lin_v_heads), 128, 0, stream_>>>(
            reinterpret_cast<float*>(core_), reinterpret_cast<const float*>(z_),
            reinterpret_cast<const float*>(g.lnorm), c_.lin_v_heads, c_.rms_eps);
        qx_valid_ = false;
        mat(g.out, core_, mix_, n, c_.lin_value_dim());
    }

    template <class Fn>
    void run_graph(std::unordered_map<uint64_t, cudaGraphExec_t>& cache, uint64_t key, Fn&& record) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            cudaGraph_t graph = nullptr;
            CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
            record();
            CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
            cudaGraphExec_t exec = nullptr;
            CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
            cudaGraphDestroy(graph);
            it = cache.emplace(key, exec).first;
        }
        CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
    }

    void attention(int l, int n, int pos, std::vector<int>& deferred, const AttnDynParams* dyn = nullptr, int fixed_splits = 0) {
        const CLayer& g = L_[l];
        const int H = c_.hidden, AD = c_.n_heads * c_.head_dim;
        const size_t kv = (size_t)slot_[l] * kv_cap_ * c_.n_kv_heads * c_.head_dim * 2;
        bool masked = false;
        if (qsa_slot_[l] >= 0) {
            const size_t ho = (size_t)qsa_slot_[l] * NMAX * H * 4;
            CUDA_CHECK(cudaMemcpyAsync(hqsa_ + ho, reinterpret_cast<void*>(xn_), (size_t)n * H * 4,
                                       cudaMemcpyDeviceToHost, stream_));
            if (!dyn && qsa_.needs_select(l, pos, n)) {
                CUDA_CHECK(cudaStreamSynchronize(stream_));
                const float* xn = reinterpret_cast<const float*>(hqsa_ + ho);
                std::vector<std::vector<int>> blocks;
                qsa_.update(l, xn, n, pos);
                qsa_.select(l, xn, n, pos, blocks);
                uint32_t* mask = reinterpret_cast<uint32_t*>(hmask_);
                for (int t = 0; t < n; t++) {
                    uint32_t* m = mask + (size_t)t * mask_words_;
                    std::fill(m, m + mask_words_, blocks[t].empty() ? 0xFFFFFFFFu : 0u);
                    for (int b : blocks[t]) m[b >> 5] |= 1u << (b & 31);
                }
                CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dmask_), hmask_, (size_t)n * mask_words_ * 4,
                                           cudaMemcpyHostToDevice, stream_));
                masked = true;
            } else if (!dyn) {
                deferred.push_back(l);
            }
        }
        mat(g.q, xn_, qg_, n, H);
        mat(g.k, xn_, kb_, n, H);
        mat(g.v, xn_, vb_, n, H);
        qk_norm_rope_kv16_kernel<<<dim3(n, c_.n_heads + c_.n_kv_heads), 256, 0, stream_>>>(
            reinterpret_cast<const float*>(qg_), reinterpret_cast<const float*>(kb_), reinterpret_cast<const float*>(vb_),
            reinterpret_cast<float*>(qo_), reinterpret_cast<uint32_t*>(kcache_ + kv),
            reinterpret_cast<uint32_t*>(vcache_ + kv), reinterpret_cast<const float*>(g.qn),
            reinterpret_cast<const float*>(g.kn), reinterpret_cast<const float*>(rope_), pos, dyn, c_.n_heads,
            c_.n_kv_heads, c_.rope_dim, c_.rms_eps);

        const int maxlen = pos + n;
        int splits = fixed_splits ? fixed_splits : (n <= NT ? (maxlen + 255) / 256 : (maxlen + 1023) / 1024);
        if (!fixed_splits) splits = std::max(1, std::min(splits, std::min(64, ATT_PART_MAX / n)));
        const int chunk = (maxlen + splits - 1) / splits;
        const float scale = 1.0f / std::sqrt((float)c_.head_dim);
        if (masked)
            attention_kv16_kernel<true><<<dim3(n, c_.n_heads, splits), 256, 0, stream_>>>(
                reinterpret_cast<const float*>(qo_), kcache_ + kv, vcache_ + kv, reinterpret_cast<float*>(part_), pos,
                c_.n_heads, c_.n_kv_heads, chunk, splits, nullptr, scale, reinterpret_cast<const uint32_t*>(dmask_),
                w_.layers[l].compress, mask_words_);
        else
            attention_kv16_kernel<false><<<dim3(n, c_.n_heads, splits), 256, 0, stream_>>>(
                reinterpret_cast<const float*>(qo_), kcache_ + kv, vcache_ + kv, reinterpret_cast<float*>(part_), pos,
                c_.n_heads, c_.n_kv_heads, chunk, splits, dyn, scale, nullptr, 0, 0);

        attn_combine_kernel<<<dim3(n, c_.n_heads), 256, 0, stream_>>>(
            reinterpret_cast<const float*>(part_), reinterpret_cast<const float*>(qg_), reinterpret_cast<float*>(att_),
            c_.n_heads, splits);
        qx_valid_ = false;
        mat(g.o, att_, mix_, n, AD);
    }

    void moe_route_and_copy(int l, int n) {
        const CLayer& g = L_[l];
        const int H = c_.hidden, TK = c_.top_k, E = c_.n_experts;
        mat(g.router, xn_, rlogits_, n, H);
        moe_route_kernel<<<n, 256, 0, stream_>>>(reinterpret_cast<const float*>(rlogits_),
                                                 reinterpret_cast<uint32_t*>(sel_), reinterpret_cast<float*>(selw_), E, TK);
        CUDA_CHECK(cudaMemcpyAsync(hsel_, reinterpret_cast<void*>(sel_), (size_t)n * TK * 4, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaMemcpyAsync(hxn_, reinterpret_cast<void*>(xn_), (size_t)n * H * 4, cudaMemcpyDeviceToHost, stream_));
    }

    void moe(int l, int n, uint64_t x_addr, bool route_already_queued = false) {
        const CLayer& g = L_[l];
        const Q4Layer& W = w_.layers[l];
        const int H = c_.hidden, TK = c_.top_k, E = c_.n_experts, I = c_.moe_inter;
        // Run router + route + host readbacks FIRST and record route_done_, then enqueue the shared expert
        // behind it so the GPU executes the shared expert while the CPU computes uncached routed experts.
        if (!route_already_queued) moe_route_and_copy(l, n);
        CUDA_CHECK(cudaEventRecord(route_done_, stream_));

        if (use_graph_ && n <= NT) {
            const uint64_t skey = ((uint64_t)l << 8) | (uint64_t)n;
            run_graph(shared_graphs_, skey, [&] {
                mat(g.sgate, xn_, sgate_, n, H);
                mat_swiglu(g.sh_gate, g.sh_up, xn_, sact_, n);
                qx_valid_ = false;
                mat(g.sh_down, sact_, sy_, n, c_.shared_inter);
            });
        } else {
            mat(g.sgate, xn_, sgate_, n, H);
            mat_swiglu(g.sh_gate, g.sh_up, xn_, sact_, n);
            qx_valid_ = false;
            mat(g.sh_down, sact_, sy_, n, c_.shared_inter);
        }

        auto tp0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 20000 && cudaEventQuery(route_done_) == cudaErrorNotReady; i++) {}
        CUDA_CHECK(cudaEventSynchronize(route_done_));
        auto tp1 = std::chrono::steady_clock::now();

        const uint32_t* sel = reinterpret_cast<const uint32_t*>(hsel_);
        if (trace_) {
            const uint32_t h[3] = {1, (uint32_t)l, (uint32_t)n};
            std::fwrite(h, 4, 3, trace_);
            std::fwrite(sel, 4, (size_t)n * TK, trace_);
        }
        per_.resize(E);
        for (auto& p : per_) p.clear();
        for (int i = 0; i < n * TK; i++) per_[sel[i]].push_back(i);

        const bool gemm = n >= EXPERT_GEMM_MIN && I % 64 == 0 && H % 64 == 0;
        const uint32_t chunk = gemm ? 64 : NT;
        ExpertEntry* ent = reinterpret_cast<ExpertEntry*>(htab_);
        uint32_t* toks = reinterpret_cast<uint32_t*>(htab_ + htab_toks_);
        uint32_t* comb = reinterpret_cast<uint32_t*>(htab_ + htab_comb_);
        uint32_t ne = 0, row = 0;
        cpu_experts_.clear();
        for (int e = 0; e < E; e++) {
            if (per_[e].empty()) continue;
            const int key = l * E + e;
            usage_[key] += per_[e].size();
            recent_[key] += (uint32_t)per_[e].size();
            if (ema_ > 0) heat_[key] += (float)per_[e].size();
            if (use_pred_) pass_used_.push_back({key, (uint32_t)per_[e].size()});
            if (where_[key].first < 0) {
                misses_ += per_[e].size();
                cpu_experts_.push_back(e);
                pending_.push_back(key);
                continue;
            }
            hits_ += per_[e].size();
            slots_[where_[key].first][where_[key].second].tick = tick_;
            const uint64_t base = slot_addr(where_[key]);
            for (size_t i0 = 0; i0 < per_[e].size(); i0 += chunk) {
                const uint32_t cnt = (uint32_t)std::min<size_t>(chunk, per_[e].size() - i0);
                ent[ne++] = {base, row, cnt};
                for (uint32_t j = 0; j < cnt; j++) {
                    const int i = per_[e][i0 + j];
                    toks[row + j] = (uint32_t)(i / TK);
                    comb[i] = row + j;
                }
                row += cnt;
            }
        }
        const uint32_t gpu_rows = row;

        // GPU: upload entries + toks and launch cached experts right away so they overlap with CPU experts
        if (ne) {
            if (fill_pending_) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_, fill_done_, 0));
                fill_pending_ = false;
            }
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dtab_), htab_, ne * sizeof(ExpertEntry),
                                       cudaMemcpyHostToDevice, stream_));
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(dtab_ + htab_toks_), htab_ + htab_toks_,
                                       gpu_rows * 4, cudaMemcpyHostToDevice, stream_));
            dispatch_expert<true>(g.cu, gemm, dtab_, dtab_ + htab_toks_, xn_, act_, 0, g.s_gate, g.off_up, g.s_up,
                                  I, H, H, I, 0, ne);
            dispatch_expert<false>(g.cd, gemm, dtab_, dtab_ + htab_toks_, act_, ey_, g.off_down, g.s_down, 0, 0,
                                   H, I, I, H, 1, ne);
        }

        // CPU: compute uncached experts directly from mmapped GGUF; pack output to fp16 for 2x less PCIe traffic
        size_t rows_cpu = 0;
        for (int e : cpu_experts_) rows_cpu += per_[e].size();
        if (rows_cpu) {
            const float* xn = reinterpret_cast<const float*>(hxn_);
            cg_.resize(rows_cpu * I);
            cu_.resize(rows_cpu * I);
            cx_.resize(rows_cpu * H);
            cy_f32_.resize(rows_cpu * H);
            cmats_.resize(cpu_experts_.size() * 3);
            std::vector<gq::Job> jobs;
            size_t r = 0;
            for (size_t a = 0; a < cpu_experts_.size(); a++) {
                const int e = cpu_experts_[a], cnt = (int)per_[e].size();
                cmats_[a * 3] = W.ex_gate.slice(e);
                cmats_[a * 3 + 1] = W.ex_up.slice(e);
                cmats_[a * 3 + 2] = W.ex_down.slice(e);
                const float* x = xn + (size_t)(per_[e][0] / TK) * H;
                if (cnt > 1) {
                    for (int j = 0; j < cnt; j++)
                        std::memcpy(&cx_[(r + j) * H], xn + (size_t)(per_[e][j] / TK) * H, (size_t)H * 4);
                    x = &cx_[r * H];
                }
                for (int j = 0; j < cnt; j++) comb[per_[e][j]] = gpu_rows + (uint32_t)(r + j);
                jobs.push_back({&cmats_[a * 3], x, &cg_[r * I], cnt});
                jobs.push_back({&cmats_[a * 3 + 1], x, &cu_[r * I], cnt});
                r += cnt;
            }
            gq::run_jobs(cpu_, jobs);
            cpu_.parallel_for((int64_t)cg_.size(), [&](int64_t b, int64_t e2) {
                for (int64_t i = b; i < e2; i++) cg_[i] = ck::silu(cg_[i]) * cu_[i];
            }, 4096);
            jobs.clear();
            r = 0;
            for (size_t a = 0; a < cpu_experts_.size(); a++) {
                const int cnt = (int)per_[cpu_experts_[a]].size();
                jobs.push_back({&cmats_[a * 3 + 2], &cg_[r * I], cy_f32_.data() + r * H, cnt});
                r += cnt;
            }
            gq::run_jobs(cpu_, jobs);
            uint16_t* cy16 = reinterpret_cast<uint16_t*>(hcy_);
            cpu_.parallel_for((int64_t)rows_cpu, [&](int64_t b, int64_t e2) {
                ck::f32_to_f16_vec(cy_f32_.data() + (size_t)b * H, cy16 + (size_t)b * H, (size_t)(e2 - b) * H);
            }, 1);
        }
        t_gpu_ += std::chrono::duration<double>(tp1 - tp0).count();
        t_cpu_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tp1).count();

        if (rows_cpu)
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(cy16_), hcy_, (size_t)rows_cpu * H * 2,
                                       cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(comb_), htab_ + htab_comb_, (size_t)n * TK * 4,
                                   cudaMemcpyHostToDevice, stream_));
        moe_combine_hc_kernel<<<dim3(groups(H, 256), n), 256, 0, stream_>>>(
            reinterpret_cast<float*>(x_addr), reinterpret_cast<const float*>(inj_),
            reinterpret_cast<const float*>(sgate_), reinterpret_cast<const float*>(sy_),
            reinterpret_cast<const float*>(ey_), reinterpret_cast<const __half*>(cy16_),
            reinterpret_cast<const uint32_t*>(comb_), reinterpret_cast<const float*>(selw_),
            H, c_.hc, TK, gpu_rows);
    }

    void forward_chunk(const int* tokens, int n, float* logits, bool spec) {
        ensure_kv(pos_ + n);
        const int H = c_.hidden, HC = c_.hc_dim();
        float* hx = reinterpret_cast<float*>(hx_);
        for (int t = 0; t < n; t++) toks_[pos_ + t] = tokens[t];
        if (trace_) {
            const uint32_t h[3] = {0, (uint32_t)pos_, (uint32_t)n};
            std::fwrite(h, 4, 3, trace_);
            std::fwrite(tokens, 4, (size_t)n, trace_);
        }
        cpu_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) {
                float* x = hx + (size_t)t * HC;
                gq::dequant_row(w_.embed.type, w_.embed.row(tokens[t]), H, x);
                for (int s = 1; s < c_.hc; s++) std::memcpy(x + (size_t)s * H, x, H * sizeof(float));
                if (ple_hist_) q4_ple_embed(c_, w_, toks_.data(), pos_ + (int)t,
                                            reinterpret_cast<float*>(hemb_) + (size_t)t * H);
            }
        }, 4);

        if (use_pred_) {
            auto tq0 = std::chrono::steady_clock::now();
            if (!pred_ready_) {
                const char* pf = std::getenv("QW_PRED_FILE");
                pred_.init(n_keys(), usage_, pf ? pf : opt_.stats_path.empty() ? "" : opt_.stats_path + ".pred");
                pred_ready_ = true;
            }
            if (pred_job_.valid()) pred_job_.get();
            t_pred_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tq0).count();
        }
        {
            const size_t nf = std::min<size_t>(MAX_FILLS, pending_.size());
            std::partial_sort(pending_.begin(), pending_.begin() + nf, pending_.end(),
                              [&](int a, int b) { return heat(a) != heat(b) ? heat(a) > heat(b) : a < b; });
            auto tf0 = std::chrono::steady_clock::now();
            size_t staging_used = 0;
            std::vector<FillJob> fjobs;
            fjobs.reserve(nf);
            for (size_t i = 0; i < nf; i++)
                if (where_[pending_[i]].first < 0) plan_fill(pending_[i], staging_used, fjobs);
            if (!fjobs.empty()) {
                // Parallel CPU repack into pinned staging, then 1 async DMA copy per expert on copy_stream_
                cpu_.parallel_for((int64_t)fjobs.size(), [&](int64_t b, int64_t e) {
                    for (int64_t j = b; j < e; j++) write_expert(fjobs[j].key, staging_ + fjobs[j].staging_off);
                });
                for (const FillJob& fj : fjobs) {
                    const Slot& sl = slots_[fj.cls][fj.slot];
                    CUDA_CHECK(cudaMemcpyAsync(sl.buf + sl.off, staging_ + fj.staging_off, fj.bytes,
                                               cudaMemcpyHostToDevice, copy_stream_));
                }
                CUDA_CHECK(cudaEventRecord(fill_done_, copy_stream_));
                fill_pending_ = true;
            }
            t_fill_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
            pending_.clear();
        }
        if (use_pred_ && !pass_used_.empty()) {
            pred_job_ = std::async(std::launch::async, [this, used = std::move(pass_used_)] { pred_.step(used); });
            pass_used_.clear();
        }
        tick_++;
        if (ema_ > 0) {
            const float k = 1.f - 1.f / ema_;
            for (float& h : heat_) h *= k;
        }

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(x_), hx_, (size_t)n * HC * 4, cudaMemcpyHostToDevice, stream_));
        if (ple_hist_)
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(emb_), hemb_, (size_t)n * H * 4, cudaMemcpyHostToDevice, stream_));

        std::vector<int> deferred;
        const bool snap = spec && n > 1;
        const int maxlen = pos_ + n;
        int attn_splits = n <= NT ? (maxlen + 255) / 256 : (maxlen + 1023) / 1024;
        attn_splits = std::max(1, std::min(attn_splits, std::min(64, ATT_PART_MAX / n)));
        if (use_graph_ && n <= NT) {
            h_attn_params_->pos = (uint32_t)pos_;
            h_attn_params_->chunk = (uint32_t)((maxlen + attn_splits - 1) / attn_splits);
            h_attn_params_->n_split = (uint32_t)attn_splits;
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_attn_params_), h_attn_params_, sizeof(AttnDynParams),
                                       cudaMemcpyHostToDevice, stream_));
        }
        for (int l = 0; l < c_.n_layers; l++) {
            const CLayer& g = L_[l];
            const bool can_graph = use_graph_ && n <= NT && (!w_.layers[l].full_attn || !qsa_.needs_select(l, pos_, n));
            if (can_graph) {
                if (w_.layers[l].full_attn && qsa_slot_[l] >= 0) deferred.push_back(l);
                const uint64_t gkey = w_.layers[l].full_attn
                                          ? ((1ull << 56) | ((uint64_t)l << 24) | ((uint64_t)n << 16) | (uint64_t)attn_splits)
                                          : ((0ull << 56) | ((uint64_t)l << 16) | ((uint64_t)n << 1) | (snap ? 1ull : 0ull));
                run_graph(pre_moe_graphs_, gkey, [&] {
                    if (w_.layers[l].ple) ple(l, n);
                    hc_mix(g.hc_attn, x_, n);
                    if (w_.layers[l].full_attn)
                        attention(l, n, pos_, deferred, reinterpret_cast<const AttnDynParams*>(d_attn_params_), attn_splits);
                    else
                        deltanet(l, n, snap);
                    hc_combine(x_, n);
                    hc_mix(g.hc_ffn, x_, n);
                    moe_route_and_copy(l, n);
                });
                moe(l, n, x_, true);
            } else {
                if (w_.layers[l].ple) ple(l, n);
                hc_mix(g.hc_attn, x_, n);
                if (w_.layers[l].full_attn) attention(l, n, pos_, deferred);
                else deltanet(l, n, snap);
                hc_combine(x_, n);
                hc_mix(g.hc_ffn, x_, n);
                moe(l, n, x_, false);
            }
        }

        const int n_out = spec ? n : 1;
        if (logits) {
            hc_mix(hc_head_, x_ + (uint64_t)(n - n_out) * HC * 4, n_out);
            mat(output_, xn_, logits_, n_out, H);
            CUDA_CHECK(cudaMemcpyAsync(hlog_, reinterpret_cast<void*>(logits_), (size_t)c_.vocab * 4 * n_out,
                                       cudaMemcpyDeviceToHost, stream_));
        }
        if (fill_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, fill_done_, 0));
            fill_pending_ = false;
        }
        for (int i = 0; i < 20000 && cudaStreamQuery(stream_) == cudaErrorNotReady; i++) {}
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (logits) std::memcpy(logits, hlog_, (size_t)c_.vocab * 4 * n_out);
        for (int l : deferred)
            qsa_.update(l, reinterpret_cast<const float*>(hqsa_ + (size_t)qsa_slot_[l] * NMAX * H * 4), n, pos_);
        pos_ += n;
    }

    void mtp_run(const int* toks, int n, int p0, float* logits, bool from_trunk, bool commit) {
        ensure_kv(std::min(max_ctx_, p0 + n + 1));
        const int H = c_.hidden, HC = c_.hc_dim(), l = c_.n_layers;
        const CLayer& g = L_[l];
        float* he = reinterpret_cast<float*>(hme_);
        cpu_.parallel_for(n, [&](int64_t b, int64_t e) {
            for (int64_t t = b; t < e; t++) gq::dequant_row(w_.embed.type, w_.embed.row(toks[t]), H, he + (size_t)t * H);
        }, 16);

        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(memb_), hme_, (size_t)n * H * 4, cudaMemcpyHostToDevice, stream_));
        if (from_trunk) {
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(mh_), hlast_, (size_t)HC * 4, cudaMemcpyDeviceToDevice, stream_));
            if (n > 1)
                CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(mh_) + (size_t)HC * 4, reinterpret_cast<void*>(x_),
                                           (size_t)(n - 1) * HC * 4, cudaMemcpyDeviceToDevice, stream_));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(mh_), reinterpret_cast<void*>(mx_), (size_t)HC * 4,
                                       cudaMemcpyDeviceToDevice, stream_));
        }
        if (commit)
            CUDA_CHECK(cudaMemcpyAsync(hlast_, reinterpret_cast<uint8_t*>(x_) + (size_t)(n - 1) * HC * 4, (size_t)HC * 4,
                                       cudaMemcpyDeviceToDevice, stream_));

        std::vector<int> deferred;
        const bool can_graph = use_graph_ && n <= NT && !qsa_.needs_select(l, p0, n);
        if (can_graph) {
            const int maxlen = p0 + n;
            int attn_splits = std::max(1, std::min((maxlen + 255) / 256, std::min(64, ATT_PART_MAX / n)));
            h_attn_params_->pos = (uint32_t)p0;
            h_attn_params_->chunk = (uint32_t)((maxlen + attn_splits - 1) / attn_splits);
            h_attn_params_->n_split = (uint32_t)attn_splits;
            CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_attn_params_), h_attn_params_, sizeof(AttnDynParams),
                                       cudaMemcpyHostToDevice, stream_));
            if (qsa_slot_[l] >= 0) deferred.push_back(l);
            const uint64_t gkey = (1ull << 56) | ((uint64_t)l << 24) | ((uint64_t)n << 16) | (uint64_t)attn_splits;
            run_graph(pre_moe_graphs_, gkey, [&] {
                add_rmsnorm_kernel<<<n, 256, 0, stream_>>>(
                    reinterpret_cast<float*>(memb_), nullptr, reinterpret_cast<const float*>(mtp_enorm_),
                    reinterpret_cast<float*>(men_), H, c_.rms_eps);
                q4_hc_norm_kernel<<<dim3(n, c_.hc), 256, 0, stream_>>>(
                    reinterpret_cast<const float*>(mh_), reinterpret_cast<const float*>(mtp_hnorm_),
                    reinterpret_cast<float*>(xnorm_), H, c_.hc, c_.rms_eps);
                q4_hc_cat_kernel<<<dim3(groups(2 * H, 256), n * c_.hc), 256, 0, stream_>>>(
                    reinterpret_cast<const float*>(men_), reinterpret_cast<const float*>(xnorm_),
                    reinterpret_cast<float*>(mcat_), H, c_.hc);
                qx_valid_ = false;
                mat(mtp_eh_, mcat_, mx_, n * c_.hc, 2 * H);
                hc_mix(g.hc_attn, mx_, n);
                attention(l, n, p0, deferred, reinterpret_cast<const AttnDynParams*>(d_attn_params_), attn_splits);
                hc_combine(mx_, n);
                hc_mix(g.hc_ffn, mx_, n);
                moe_route_and_copy(l, n);
            });
            moe(l, n, mx_, true);
        } else {
            add_rmsnorm_kernel<<<n, 256, 0, stream_>>>(
                reinterpret_cast<float*>(memb_), nullptr, reinterpret_cast<const float*>(mtp_enorm_),
                reinterpret_cast<float*>(men_), H, c_.rms_eps);
            q4_hc_norm_kernel<<<dim3(n, c_.hc), 256, 0, stream_>>>(
                reinterpret_cast<const float*>(mh_), reinterpret_cast<const float*>(mtp_hnorm_),
                reinterpret_cast<float*>(xnorm_), H, c_.hc, c_.rms_eps);
            q4_hc_cat_kernel<<<dim3(groups(2 * H, 256), n * c_.hc), 256, 0, stream_>>>(
                reinterpret_cast<const float*>(men_), reinterpret_cast<const float*>(xnorm_),
                reinterpret_cast<float*>(mcat_), H, c_.hc);
            qx_valid_ = false;
            mat(mtp_eh_, mcat_, mx_, n * c_.hc, 2 * H);
            hc_mix(g.hc_attn, mx_, n);
            attention(l, n, p0, deferred);
            hc_combine(mx_, n);
            hc_mix(g.hc_ffn, mx_, n);
            moe(l, n, mx_, false);
        }

        if (logits) {
            hc_mix(mtp_head_, mx_ + (uint64_t)(n - 1) * HC * 4, 1);
            mat(output_, xn_, logits_, 1, H);
            CUDA_CHECK(cudaMemcpyAsync(hlog_, reinterpret_cast<void*>(logits_), (size_t)c_.vocab * 4,
                                       cudaMemcpyDeviceToHost, stream_));
        }
        if (fill_pending_) {
            CUDA_CHECK(cudaStreamWaitEvent(stream_, fill_done_, 0));
            fill_pending_ = false;
        }
        for (int i = 0; i < 20000 && cudaStreamQuery(stream_) == cudaErrorNotReady; i++) {}
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        if (logits) std::memcpy(logits, hlog_, (size_t)c_.vocab * 4);
        for (int dl : deferred)
            qsa_.update(dl, reinterpret_cast<const float*>(hqsa_ + (size_t)qsa_slot_[dl] * NMAX * H * 4), n, p0);
    }

    const Q4Config& c_;
    const Q4Weights& w_;
    int max_ctx_;
    size_t sys_free0_ = 0, kv_bytes_ = 0;
    int kv_cap_ = 0, n_kv_slots_ = 0;
    VulkanOptions opt_;
    int pos_ = 0, saved_pos_ = 0;

    cudaStream_t stream_ = nullptr, copy_stream_ = nullptr;
    cudaEvent_t route_done_ = nullptr, fill_done_ = nullptr;
    CudaArena arena_;
    std::vector<void*> dev_bufs_, host_bufs_;
    std::unordered_map<void*, size_t> dev_sizes_;
    size_t dev_alloc_bytes_ = 0;

    ThreadPool cpu_;
    QsaIndexer qsa_;

    std::vector<CLayer> L_;
    CM output_;
    CHc hc_head_;

    int draft_ = 0, spec_pos0_ = 0;
    std::vector<int> spec_toks_;
    std::vector<float> mlogits_;
    CM mtp_eh_;
    CHc mtp_head_;
    uint64_t mtp_enorm_ = 0, mtp_hnorm_ = 0, mx_ = 0, mh_ = 0, mcat_ = 0, men_ = 0, memb_ = 0;
    uint8_t* hme_ = nullptr;
    uint8_t* hlast_ = nullptr;
    uint8_t* hlast_saved_ = nullptr;
    size_t hlast_sz_ = 0;
    uint8_t* conv_snap_ = nullptr;
    uint8_t* rec_snap_ = nullptr;
    uint64_t rope_ = 0;
    std::vector<int> slot_, qsa_slot_, toks_;
    int ple_hist_ = 0, mask_words_ = 1;

    // activations
    uint64_t x_ = 0, xnorm_ = 0, gate_ = 0, lo_ = 0, inj_ = 0, xn_ = 0, mix_ = 0;
    uint64_t qkv_ = 0, z_ = 0, a_ = 0, b_ = 0, core_ = 0;
    uint64_t qg_ = 0, kb_ = 0, vb_ = 0, qo_ = 0, att_ = 0, part_ = 0;
    uint64_t sact_ = 0, sy_ = 0, sgate_ = 0, act_ = 0, ey_ = 0, cy16_ = 0;
    uint64_t rlogits_ = 0, selw_ = 0, comb_ = 0, sel_ = 0, logits_ = 0, dmask_ = 0, dtab_ = 0, xq_ = 0, xs_ = 0;
    uint64_t d_attn_params_ = 0;
    AttnDynParams* h_attn_params_ = nullptr;
    uint64_t emb_ = 0, pkey_ = 0, pval_ = 0, pgated_ = 0, ppad_ = 0;
    bool fill_pending_ = false, use_dp4a_ = false, use_graph_ = true, qx_valid_ = false;
    uint64_t qx_x_ = 0;
    int qx_n_ = 0, qx_cols_ = 0, qx_stride_ = 0;
    std::unordered_map<uint64_t, cudaGraphExec_t> pre_moe_graphs_, shared_graphs_;

    uint8_t *kcache_ = nullptr, *vcache_ = nullptr;
    uint8_t *conv_state_ = nullptr, *rec_state_ = nullptr, *ple_state_ = nullptr;
    uint8_t *conv_saved_ = nullptr, *rec_saved_ = nullptr, *ple_saved_ = nullptr;
    size_t conv_state_sz_ = 0, rec_state_sz_ = 0, ple_state_sz_ = 0;

    uint8_t *hx_ = nullptr, *hemb_ = nullptr, *hsel_ = nullptr, *hlog_ = nullptr;
    uint8_t *hqsa_ = nullptr, *hmask_ = nullptr, *hxn_ = nullptr, *htab_ = nullptr, *hcy_ = nullptr, *staging_ = nullptr;
    size_t htab_toks_ = 0, htab_comb_ = 0;

    // experts
    struct Slot { uint8_t* buf = nullptr; size_t off = 0; int key = -1; uint64_t tick = 0; };
    std::vector<std::vector<Slot>> slots_;
    std::vector<size_t> cls_size_;
    std::vector<std::pair<int, int>> where_;
    std::vector<int> pending_;
    std::vector<std::vector<int>> per_;
    std::vector<int> cpu_experts_;
    std::vector<float> cg_, cu_, cx_, cy_f32_;
    std::vector<gq::GMat> cmats_;
    std::vector<uint64_t> usage_;
    std::vector<uint32_t> recent_;
    std::vector<float> heat_;
    float ema_ = 0, ema_margin_ = 2.5f;
    ExpertPredictor pred_;
    bool use_pred_ = false, pred_ready_ = false;
    float pred_margin_ = 1.f;
    double t_pred_ = 0;
    std::future<void> pred_job_;
    std::vector<std::pair<int, uint32_t>> pass_used_;
    float heat(int key) const { return use_pred_ ? pred_.score(key) : ema_ > 0 ? heat_[key] : (float)recent_[key]; }
    float fill_margin() const { return use_pred_ ? pred_margin_ : ema_ > 0 ? ema_margin_ : (float)FILL_MARGIN; }
    FILE* trace_ = nullptr;
    double t_gpu_ = 0, t_cpu_ = 0, t_fill_ = 0;
    uint64_t tick_ = 0, hits_ = 0, misses_ = 0, fills_ = 0;
};

}  // namespace

std::unique_ptr<Backend> make_qwen4_cuda_backend(const Q4Config& cfg, const Q4Weights& w, int max_ctx,
                                                 const VulkanOptions& opt) {
    return std::make_unique<Qwen4Cuda>(cfg, w, max_ctx, opt);
}
