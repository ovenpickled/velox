#pragma once
#include <cstddef>

namespace ops {

void matmul(float* out, const float* a, const float* b, int M, int N, int K);
void rmsnorm(float* out, const float* x, const float* weight, int size, float eps);
void rope(float* q, float* k, int head_dim, int num_q_heads, int num_kv_heads, int pos, float theta);
void silu(float* x, int size);
void multiply(float* out, const float* a, const float* b, int size);
void add(float* out, const float* a, const float* b, int size);
void add_bias(float* x, const float* bias, int size);
void softmax(float* x, int size);
void embedding_lookup(float* out, const float* table, int token_id, int dim);

}
