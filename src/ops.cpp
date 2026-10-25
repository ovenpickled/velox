#include "ops.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace ops {

void quantize_rowwise(Int8Tensor& out, const float* weights, int M, int K) {
    out.M = M;
    out.K = K;
    out.data.resize(static_cast<size_t>(M) * K);
    out.scales.resize(M);

    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        float max_val = 0.0f;
        for (int j = 0; j < K; j++) {
            float val = std::abs(weights[static_cast<size_t>(i) * K + j]);
            if (val > max_val) max_val = val;
        }

        float scale = max_val / 127.0f;
        out.scales[i] = scale;

        float inv_scale = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
        for (int j = 0; j < K; j++) {
            float scaled = weights[static_cast<size_t>(i) * K + j] * inv_scale;
            int8_t q = static_cast<int8_t>(std::round(std::max(-127.0f, std::min(127.0f, scaled))));
            out.data[static_cast<size_t>(i) * K + j] = q;
        }
    }
}

#ifndef ACT_GROUP
#define ACT_GROUP 64
#endif

// Diagnostic only: INT8 weights, FP32 activations (no activation quantization)
static void matmul_w8a32(float* out, const float* a, const Int8Tensor& w, int M, int N, int K) {
    const int8_t* wd = w.data.data();
    const float* ws = w.scales.data();
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        const int8_t* row_w = wd + static_cast<size_t>(j) * K;
        for (int i = 0; i < M; i++) {
            const float* row_a = a + static_cast<size_t>(i) * K;
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++) sum += row_a[k] * static_cast<float>(row_w[k]);
            out[static_cast<size_t>(i) * N + j] = sum * ws[j];
        }
    }
}

void matmul_w8a8(float* out, const float* a, const Int8Tensor& w, int M, int N, int K) {
    static const bool w8a32 = [] {
        bool on = std::getenv("VELOX_W8A32") != nullptr;
        if (on) std::fprintf(stderr, "[diag] W8A32 path active\n");
        return on;
    }();
    if (w8a32) { matmul_w8a32(out, a, w, M, N, K); return; }

    const int G = (K + ACT_GROUP - 1) / ACT_GROUP;   // activation groups per row
    static thread_local std::vector<int8_t> a_q_buf;
    static thread_local std::vector<float> a_scales_buf;
    if (a_q_buf.size() < static_cast<size_t>(M) * K) a_q_buf.resize(static_cast<size_t>(M) * K);
    if (a_scales_buf.size() < static_cast<size_t>(M) * G) a_scales_buf.resize(static_cast<size_t>(M) * G);
    int8_t* a_q = a_q_buf.data();
    float* a_scales = a_scales_buf.data();

    // 1. Quantize activations: one scale per group of ACT_GROUP channels
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        const float* src = a + static_cast<size_t>(i) * K;
        for (int g = 0; g < G; g++) {
            const int k0 = g * ACT_GROUP, k1 = std::min(k0 + ACT_GROUP, K);
            float max_val = 0.0f;
            for (int k = k0; k < k1; k++) max_val = std::max(max_val, std::abs(src[k]));
            const float scale = max_val / 127.0f;
            a_scales[static_cast<size_t>(i) * G + g] = scale;
            const float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
            for (int k = k0; k < k1; k++) {
                float s = src[k] * inv;
                a_q[static_cast<size_t>(i) * K + k] =
                    static_cast<int8_t>(std::round(std::max(-127.0f, std::min(127.0f, s))));
            }
        }
    }

    // 2. Matmul: int32 dot product per group, rescaled by that group's scale
    const int8_t* wd = w.data.data();
    const float* ws = w.scales.data();
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        const int8_t* row_w = wd + static_cast<size_t>(j) * K;
        const float wscale = ws[j];
        for (int i = 0; i < M; i++) {
            const int8_t* row_a = a_q + static_cast<size_t>(i) * K;
            const float* sc = a_scales + static_cast<size_t>(i) * G;
            float acc = 0.0f;
            for (int g = 0; g < G; g++) {
                const int k0 = g * ACT_GROUP, k1 = std::min(k0 + ACT_GROUP, K);
                int32_t s = 0;
                for (int k = k0; k < k1; k++) {
                    s += static_cast<int32_t>(row_a[k]) * static_cast<int32_t>(row_w[k]);
                }
                acc += static_cast<float>(s) * sc[g];
            }
            out[static_cast<size_t>(i) * N + j] = acc * wscale;
        }
    }
}

void matmul(float* out, const float* a, const float* b, int M, int N, int K) {
    // a: [M, K]   b: [N, K]   out: [M, N]. Parallel over N; the simd reduction
    // lets the compiler vectorize the dot product without -ffast-math.
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j++) {
        const float* row_b = b + static_cast<size_t>(j) * K;
        for (int i = 0; i < M; i++) {
            const float* row_a = a + static_cast<size_t>(i) * K;
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; ++k) {
                sum += row_a[k] * row_b[k];
            }
            out[static_cast<size_t>(i) * N + j] = sum;
        }
    }
}

void rmsnorm(float* out, const float* x, const float* weight, int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    float mean = sum_sq / size;
    float inv_std = 1.0f / std::sqrt(mean + eps);
    for (int i = 0; i < size; ++i) {
        out[i] = x[i] * inv_std * weight[i];
    }
}

void rope(float* q, float* k, int head_dim, int num_q_heads, int num_kv_heads, int pos, float theta) {
    int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        float freq = 1.0f / std::pow(theta, (2.0f * i) / head_dim);
        float angle = pos * freq;
        float cos_val = std::cos(angle);
        float sin_val = std::sin(angle);

        for (int h = 0; h < num_q_heads; ++h) {
            int idx0 = h * head_dim + i;
            int idx1 = h * head_dim + i + half_dim;
            float q0 = q[idx0];
            float q1 = q[idx1];
            q[idx0] = q0 * cos_val - q1 * sin_val;
            q[idx1] = q0 * sin_val + q1 * cos_val;
        }

        for (int h = 0; h < num_kv_heads; ++h) {
            int idx0 = h * head_dim + i;
            int idx1 = h * head_dim + i + half_dim;
            float k0 = k[idx0];
            float k1 = k[idx1];
            k[idx0] = k0 * cos_val - k1 * sin_val;
            k[idx1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

void silu(float* x, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void multiply(float* out, const float* a, const float* b, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = a[i] * b[i];
    }
}

void add(float* out, const float* a, const float* b, int size) {
    for (int i = 0; i < size; ++i) {
        out[i] = a[i] + b[i];
    }
}

void add_bias(float* x, const float* bias, int size) {
    for (int i = 0; i < size; ++i) {
        x[i] += bias[i];
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

void embedding_lookup(float* out, const float* table, int token_id, int dim) {
    std::memcpy(out, table + static_cast<size_t>(token_id) * dim, dim * sizeof(float));
}

}
