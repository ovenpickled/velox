// Quick vectorized FP32 throughput test using SIMD reduction (like our matmul)
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>
#include <omp.h>

int main() {
    // Simulate our matmul's inner loop: dot product with #pragma omp simd
    const int K = 896;
    const int N = 151936;  // LM head rows
    const int reps = 5;

    std::vector<float> a(K), b(N * K);
    for (int i = 0; i < K; i++) a[i] = 0.001f * i;
    for (size_t i = 0; i < b.size(); i++) b[i] = 0.001f * (i % 1000);
    std::vector<float> out(N);

    printf("=== Vectorized FP32 Matmul Throughput (M=1, N=%d, K=%d) ===\n", N, K);

    std::vector<double> gflops_list;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; j++) {
            const float* row_b = b.data() + (size_t)j * K;
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < K; k++) sum += a[k] * row_b[k];
            out[j] = sum;
        }

        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        double flops = 2.0 * N * K;
        gflops_list.push_back(flops / dt / 1e9);
    }

    std::sort(gflops_list.begin(), gflops_list.end());
    printf("  FP32 dot (simd):  %.2f GFLOP/s (median)\n", gflops_list[reps/2]);
    printf("  Effective BW:     %.2f GB/s (if memory-bound, bytes = N*K*4)\n",
           (double)N * K * 4.0 / 1e9 / (2.0 * N * K / gflops_list[reps/2] / 1e9));

    // INT8 version — same pattern as matmul_w8a8
    std::vector<int8_t> ai(K), bi(N * K);
    for (int i = 0; i < K; i++) ai[i] = (int8_t)(i % 127);
    for (size_t i = 0; i < bi.size(); i++) bi[i] = (int8_t)(i % 127 - 63);
    std::vector<float> ws(N, 0.01f);

    printf("\n=== Vectorized INT8 Matmul Throughput (M=1, N=%d, K=%d) ===\n", N, K);

    std::vector<double> int8_gops;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; j++) {
            const int8_t* row_w = bi.data() + (size_t)j * K;
            int32_t s = 0;
            for (int k = 0; k < K; k++) {
                s += (int32_t)ai[k] * (int32_t)row_w[k];
            }
            out[j] = (float)s * ws[j];
        }

        auto t1 = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        double ops = 2.0 * N * K;
        int8_gops.push_back(ops / dt / 1e9);
    }

    std::sort(int8_gops.begin(), int8_gops.end());
    printf("  INT8 dot:         %.2f GOP/s (median)\n", int8_gops[reps/2]);
    printf("  Effective BW:     %.2f GB/s (bytes = N*K*1)\n",
           (double)N * K * 1.0 / 1e9 / (2.0 * N * K / int8_gops[reps/2] / 1e9));

    // prevent optimization
    volatile float v = 0;
    for (int i = 0; i < N; i++) v += out[i];

    return 0;
}
