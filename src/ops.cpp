#include "ops.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <omp.h>

namespace ops {

void matmul(float* out, const float* a, const float* b, int M, int N, int K) {
    #pragma omp parallel for
    for (int i0 = 0; i0 < M; i0 += 32) {
        for (int j0 = 0; j0 < N; j0 += 32) {
            for (int i = i0; i < std::min(i0 + 32, M); ++i) {
                for (int j = j0; j < std::min(j0 + 32, N); ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += a[i * K + k] * b[j * K + k];
                    }
                    out[i * N + j] = sum;
                }
            }
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
    std::memcpy(out, table + token_id * dim, dim * sizeof(float));
}

}
