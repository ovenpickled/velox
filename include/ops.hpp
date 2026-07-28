#pragma once
#include <cstddef>
#include "tensor.hpp"

namespace ops {

void quantize_rowwise(Int8Tensor& out, const float* weights, int M, int K);
void matmul_w8a8(float* out, const float* a, const Int8Tensor& w, int M, int N, int K);

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
