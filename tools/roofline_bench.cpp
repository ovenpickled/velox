// roofline_bench.cpp — Measure peak memory bandwidth and peak arithmetic throughput
// to construct a roofline model for the machine.
//
// Build:  g++ -O3 -march=native -fopenmp -o roofline_bench tools/roofline_bench.cpp -lm
// Run:    ./roofline_bench
//
// Measures:
//   1. Memory bandwidth (STREAM-style Copy, Scale, Add, Triad)
//   2. Peak FP32 throughput (FMA-dominated loop)
//   3. Peak INT8 dot-product throughput (simulates W8A8 inner loop)

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static double now_sec() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// 1. STREAM bandwidth benchmark
// ---------------------------------------------------------------------------
// Uses arrays large enough to exceed L3 cache.  Each kernel is run
// `reps` times; we take the median bandwidth.

static void stream_benchmark(size_t n_elements, int reps) {
    // Allocate aligned arrays
    std::vector<double> a(n_elements), b(n_elements), c(n_elements);

    // Initialize
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_elements; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    double scalar = 3.0;

    auto bench = [&](const char* name, int bytes_per_elem, auto fn) {
        std::vector<double> bws;
        for (int r = 0; r < reps; r++) {
            double t0 = now_sec();
            fn();
            double t1 = now_sec();
            double bytes = (double)n_elements * bytes_per_elem;
            bws.push_back(bytes / (t1 - t0));
        }
        double med = median(bws) / 1e9;
        printf("  %-8s  %7.2f GB/s\n", name, med);
        return med;
    };

    printf("=== STREAM Bandwidth (N = %zu doubles, %.1f MiB per array) ===\n",
           n_elements, n_elements * 8.0 / (1024*1024));

    double bw_copy = bench("Copy", 2*8, [&]{
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_elements; i++) c[i] = a[i];
    });

    double bw_scale = bench("Scale", 2*8, [&]{
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_elements; i++) b[i] = scalar * c[i];
    });

    double bw_add = bench("Add", 3*8, [&]{
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_elements; i++) c[i] = a[i] + b[i];
    });

    double bw_triad = bench("Triad", 3*8, [&]{
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_elements; i++) a[i] = b[i] + scalar * c[i];
    });

    printf("  -------\n");
    printf("  Best:    %7.2f GB/s (Triad)\n", bw_triad);
    printf("\n");
}

// ---------------------------------------------------------------------------
// 2. Peak FP32 arithmetic throughput (FMA chains)
// ---------------------------------------------------------------------------
// Each thread runs independent FMA chains to saturate FP execution units.
// We use multiple accumulators to avoid latency bottlenecks.

static void fp32_throughput_benchmark(int reps) {
    int nthreads = omp_get_max_threads();
    // Each iteration does 16 FMAs (a = a * b + c counts as 2 FLOPs)
    const long iters = 100000000L;

    printf("=== Peak FP32 Throughput (%d threads, %ld iterations/thread) ===\n", nthreads, iters);

    std::vector<double> gflops_list;
    for (int r = 0; r < reps; r++) {
        volatile float sink = 0;  // prevent dead-code elimination
        double t0 = now_sec();

        #pragma omp parallel
        {
            // 8 independent FMA accumulators to hide latency
            float a0 = 1.0001f, a1 = 1.0002f, a2 = 1.0003f, a3 = 1.0004f;
            float a4 = 1.0005f, a5 = 1.0006f, a6 = 1.0007f, a7 = 1.0008f;
            float b = 1.000001f, c = 0.000001f;

            for (long i = 0; i < iters; i++) {
                a0 = a0 * b + c;  a1 = a1 * b + c;
                a2 = a2 * b + c;  a3 = a3 * b + c;
                a4 = a4 * b + c;  a5 = a5 * b + c;
                a6 = a6 * b + c;  a7 = a7 * b + c;
            }
            // Use the results so the compiler can't optimize them away
            float s = a0+a1+a2+a3+a4+a5+a6+a7;
            #pragma omp atomic
            sink += s;
        }

        double t1 = now_sec();
        // 8 FMAs per iteration, 2 FLOPs per FMA = 16 FLOPs/iter
        double flops = (double)nthreads * iters * 8.0 * 2.0;
        gflops_list.push_back(flops / (t1 - t0) / 1e9);
    }

    double med = median(gflops_list);
    printf("  Scalar FMA:  %7.2f GFLOP/s\n", med);
    printf("\n");
}

// ---------------------------------------------------------------------------
// 3. Peak INT8 dot product throughput
// ---------------------------------------------------------------------------
// Simulates the W8A8 inner kernel: int8×int8 → int32 accumulate.
// This is what the matmul_w8a8 inner loop does.

static void int8_throughput_benchmark(int reps) {
    int nthreads = omp_get_max_threads();
    // Simulate dot products of length K=896 (Qwen2.5-0.5B hidden size)
    const int K = 896;
    const int num_dots = 200000;

    printf("=== INT8 Dot Product Throughput (%d threads, %d dots × K=%d) ===\n",
           nthreads, num_dots, K);

    // Prepare data
    std::vector<int8_t> a_data(K * num_dots), b_data(K * num_dots);
    for (size_t i = 0; i < a_data.size(); i++) {
        a_data[i] = (int8_t)((i * 7 + 3) % 255 - 127);
        b_data[i] = (int8_t)((i * 11 + 5) % 255 - 127);
    }
    std::vector<int32_t> results(num_dots);

    std::vector<double> gops_list;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();

        #pragma omp parallel for schedule(static)
        for (int d = 0; d < num_dots; d++) {
            const int8_t* pa = a_data.data() + (size_t)d * K;
            const int8_t* pb = b_data.data() + (size_t)d * K;
            int32_t s = 0;
            for (int k = 0; k < K; k++) {
                s += (int32_t)pa[k] * (int32_t)pb[k];
            }
            results[d] = s;
        }

        double t1 = now_sec();
        // 2 ops per multiply-accumulate (mul + add)
        double ops = (double)num_dots * K * 2.0;
        gops_list.push_back(ops / (t1 - t0) / 1e9);
    }

    double med = median(gops_list);
    printf("  INT8 MAC:    %7.2f GOP/s\n", med);

    // Prevent optimizing away
    volatile int32_t v = 0;
    for (int i = 0; i < num_dots; i++) v += results[i];

    printf("\n");
}

// ---------------------------------------------------------------------------
// 4. Effective bandwidth of a single matmul_w8a8-like kernel
// ---------------------------------------------------------------------------
// Simulates the memory access pattern of decode (M=1, N=151936, K=896) to
// measure achieved bandwidth for the LM head, which is the biggest single
// weight matrix.

static void lm_head_bandwidth_benchmark(int reps) {
    const int M = 1;
    const int N = 151936;  // vocab_size
    const int K = 896;     // hidden_size

    printf("=== LM Head Effective Bandwidth (M=%d, N=%d, K=%d) ===\n", M, N, K);

    // Allocate INT8 weights + FP32 scales (simulating Int8Tensor layout)
    std::vector<int8_t> w(N * K);
    std::vector<float> ws(N);
    std::vector<int8_t> a(M * K);
    std::vector<float> out(M * N);

    for (size_t i = 0; i < w.size(); i++) w[i] = (int8_t)(i % 255 - 127);
    for (int i = 0; i < N; i++) ws[i] = 0.01f;
    for (int i = 0; i < K; i++) a[i] = (int8_t)(i % 127);

    std::vector<double> bws;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < N; j++) {
            const int8_t* row_w = w.data() + (size_t)j * K;
            int32_t s = 0;
            for (int k = 0; k < K; k++) {
                s += (int32_t)a[k] * (int32_t)row_w[k];
            }
            out[j] = (float)s * ws[j];
        }

        double t1 = now_sec();
        // Bytes read: N*K (weights) + N*4 (scales) + M*K (activations, but tiny)
        // Bytes written: M*N*4 (output)
        double bytes = (double)N * K + (double)N * 4 + (double)M * K + (double)M * N * 4;
        bws.push_back(bytes / (t1 - t0));
    }

    double med = median(bws) / 1e9;
    printf("  Achieved:   %7.2f GB/s\n", med);
    printf("  Weight data: %.1f MiB (INT8)\n", (double)N * K / (1024*1024));

    // Prevent optimizing away
    volatile float v = 0;
    for (int i = 0; i < N; i++) v += out[i];

    printf("\n");
}

// ---------------------------------------------------------------------------
// 5. Roofline summary
// ---------------------------------------------------------------------------

static void print_roofline_summary(double peak_bw_gbs, double peak_gflops) {
    printf("=== Roofline Summary ===\n");
    printf("  Peak bandwidth:  %.2f GB/s\n", peak_bw_gbs);
    printf("  Peak compute:    %.2f GFLOP/s (scalar FMA)\n", peak_gflops);
    printf("  Ridge point:     %.2f FLOP/byte\n", peak_gflops / peak_bw_gbs);
    printf("\n");

    // Compute operational intensity for decode matmuls
    // For M=1 (decode), a matmul [1,K] × [N,K]^T reads N*sizeof_w bytes
    // and does 2*N*K FLOPs, so OI = 2*K / sizeof_w

    printf("  --- Operational Intensity for decode (M=1) ---\n");
    printf("  %-28s  OI (FLOP/B)  Bound\n", "Kernel");
    printf("  %-28s  ----------  -----\n", "------");

    auto classify = [&](const char* name, int K, int elem_bytes) {
        double oi = 2.0 * K / elem_bytes;
        double ridge = peak_gflops / peak_bw_gbs;
        const char* bound = (oi < ridge) ? "MEM" : "COMPUTE";
        printf("  %-28s  %10.1f  %s\n", name, oi, bound);
    };

    // Qwen2.5-0.5B dimensions
    classify("W8A8 layer (K=896)",        896, 1);  // INT8 weights
    classify("FP32 layer (K=896)",        896, 4);  // FP32 weights
    classify("W8A8 LM head (K=896)",      896, 1);  // INT8 LM head
    classify("FP32 LM head (K=896)",      896, 4);  // FP32 LM head
    classify("W8A8 MLP gate (K=896)",     896, 1);  // gate_proj
    classify("W8A8 MLP down (K=2432)",   2432, 1);  // down_proj (intermediate)

    printf("\n");
    printf("  Note: OI for W8A8 counts only weight bytes (dominant at M=1).\n");
    printf("  At M=1, every kernel is deep in the memory-bound regime\n");
    printf("  unless OI > %.1f FLOP/byte (the ridge point).\n", peak_gflops / peak_bw_gbs);
    printf("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    printf("Roofline Benchmark — %d threads (OMP_NUM_THREADS=%s)\n\n",
           omp_get_max_threads(),
           getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "default");

    // 1. STREAM — use 64 MiB per array to exceed L3
    size_t stream_n = 8 * 1024 * 1024;  // 8M doubles = 64 MiB
    stream_benchmark(stream_n, 20);

    // 2. FP32 peak
    fp32_throughput_benchmark(10);

    // 3. INT8 dot product
    int8_throughput_benchmark(10);

    // 4. LM head effective bandwidth
    lm_head_bandwidth_benchmark(20);

    // 5. Summary — use Triad BW and scalar FMA peak
    // (We'll read the numbers from the output and plug them in;
    //  for now, re-run the key benchmarks to get the values)
    printf("(Run complete. Use the Triad BW and scalar FMA GFLOP/s above\n");
    printf(" to compute your roofline ridge point.)\n\n");

    return 0;
}
