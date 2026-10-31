// roofline_bench.cpp — Measure peak memory bandwidth and peak arithmetic throughput
// to construct a roofline model for the machine.
//
// Build:  g++ -O3 -march=native -fopenmp -o roofline_bench tools/roofline_bench.cpp -lm
// Run:    ./roofline_bench
//
// Measures:
//   1. Memory bandwidth (STREAM-style Copy, Scale, Add, Triad, and Read-only)
//   2. Peak FP32 throughput (vector FMA)
//   3. Peak INT8 dot-product throughput (in-cache compute ceiling)

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

static double stream_benchmark(size_t n_elements, int reps) {
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
            bws.push_back(bytes / (t1 - t0) / 1e9);
        }
        double med = median(bws);
        double mn = *std::min_element(bws.begin(), bws.end());
        double mx = *std::max_element(bws.begin(), bws.end());
        printf("  %-8s  %7.2f GB/s  (min: %5.2f, max: %5.2f)\n", name, med, mn, mx);
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
    return bw_triad;
}

// ---------------------------------------------------------------------------
// Read-only bandwidth benchmark
// ---------------------------------------------------------------------------
static double read_bandwidth_benchmark(size_t n_floats, int reps) {   // e.g. 64M uint32_t = 256 MiB
    std::vector<uint32_t> data(n_floats);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n_floats; i++) data[i] = 1u;
    std::vector<double> bws;
    volatile uint32_t sink = 0;
    for (int r = 0; r < reps; r++) {
        uint32_t total = 0u;
        double t0 = now_sec();
        #pragma omp parallel for simd schedule(static) reduction(+:total)
        for (size_t i = 0; i < n_floats; i++) total += data[i];
        double t1 = now_sec();
        sink += total;
        bws.push_back((double)n_floats * 4 / (t1 - t0) / 1e9);
    }
    double med = median(bws);
    double mn = *std::min_element(bws.begin(), bws.end());
    double mx = *std::max_element(bws.begin(), bws.end());
    printf("  Read-only:  %7.2f GB/s  (min: %5.2f, max: %5.2f)\n", med, mn, mx);
    return med;
}

// ---------------------------------------------------------------------------
// 2. Peak FP32 arithmetic throughput (vector FMA)
// ---------------------------------------------------------------------------
static double fp32_vector_benchmark(int reps) {
    const long ITERS = 20000000L;
    const int nthreads = omp_get_max_threads();
    volatile float sink = 0;
    std::vector<double> gflops;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        #pragma omp parallel
        {
            alignas(64) float acc[8][16];                 // 8 independent chains of 16 floats
            for (int i = 0; i < 8; i++) for (int l = 0; l < 16; l++) acc[i][l] = 1.0f + 0.001f * (i * 16 + l);
            const float b = 1.0000001f, c = 1e-7f;
            for (long it = 0; it < ITERS; it++)
                for (int i = 0; i < 8; i++) {
                    #pragma omp simd
                    for (int l = 0; l < 16; l++) acc[i][l] = __builtin_fmaf(acc[i][l], b, c);
                }
            float s = 0;
            for (int i = 0; i < 8; i++) for (int l = 0; l < 16; l++) s += acc[i][l];
            #pragma omp critical
            sink += s;
        }
        double t1 = now_sec();
        gflops.push_back((double)nthreads * ITERS * 8 * 16 * 2.0 / (t1 - t0) / 1e9);
    }
    double med = median(gflops);
    printf("  FP32 vector FMA: %7.2f GFLOP/s\n", med);
    return med;
}

// ---------------------------------------------------------------------------
// 3. Peak INT8 compute throughput (in-cache)
// ---------------------------------------------------------------------------
static double int8_compute_benchmark(int reps) {
    const int K = 896, ROWS = 16;          // 16 x 896 B = 14 KB of weights per thread: stays in L1
    const long REPEAT = 400000;            // tune so one repetition takes at least ~0.2 s
    const int nthreads = omp_get_max_threads();
    volatile int32_t sink = 0;
    std::vector<double> gops;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        #pragma omp parallel
        {
            std::vector<int8_t> w(ROWS * K), a(K);
            for (int i = 0; i < ROWS * K; i++) w[i] = (int8_t)((i * 7 + 3) % 251 - 125);
            for (int i = 0; i < K; i++)        a[i] = (int8_t)((i * 11 + 5) % 251 - 125);
            int32_t total = 0;
            for (long rep = 0; rep < REPEAT; rep++) {
                int32_t round_sum = 0;
                for (int row = 0; row < ROWS; row++) {
                    const int8_t* pw = w.data() + (size_t)row * K;
                    int32_t s = 0;
                    for (int k = 0; k < K; k++) s += (int32_t)a[k] * (int32_t)pw[k];
                    round_sum += s;
                }
                a[0] = (int8_t)(round_sum & 0x3f);   // dependency on the previous round: stops the compiler hoisting the work
                total += round_sum;
            }
            #pragma omp critical
            sink += total;
        }
        double t1 = now_sec();
        gops.push_back((double)nthreads * REPEAT * ROWS * K * 2.0 / (t1 - t0) / 1e9);
    }
    double med = median(gops);
    double mn = *std::min_element(gops.begin(), gops.end());
    double mx = *std::max_element(gops.begin(), gops.end());
    printf("  INT8 compute (in-cache, original):          %7.2f GOP/s  (min: %5.2f, max: %5.2f)\n", med, mn, mx);
    return med;
}

static double int8_compute_variant_g(int reps) {
    const int K = 896, ROWS = 16;
    const long REPEAT = 200000;
    const int nthreads = omp_get_max_threads();
    volatile int32_t sink = 0;
    std::vector<double> gops;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        #pragma omp parallel
        {
            std::vector<int8_t> w(ROWS * K), a(K);
            for (int i = 0; i < ROWS * K; i++) w[i] = (int8_t)((i * 7 + 3) % 251 - 125);
            for (int i = 0; i < K; i++)        a[i] = (int8_t)((i * 11 + 5) % 251 - 125);
            int32_t total = 0;
            for (long rep = 0; rep < REPEAT; rep++) {
                int32_t round_sum = 0;
                for (int row = 0; row < ROWS; row++) {
                    float acc = 0.0f;
                    for (int g = 0; g < K / 64; g++) {
                        int32_t s = 0;
                        const int8_t* pa = a.data() + g * 64;
                        const int8_t* pw = w.data() + (size_t)row * K + g * 64;
                        for (int k = 0; k < 64; k++) s += (int32_t)pa[k] * (int32_t)pw[k];
                        acc += (float)s * 0.01f;
                    }
                    round_sum += (int32_t)acc;
                }
                a[0] = (int8_t)(round_sum & 0x3f);
                total += round_sum;
            }
            #pragma omp critical
            sink += total;
        }
        double t1 = now_sec();
        gops.push_back((double)nthreads * REPEAT * ROWS * K * 2.0 / (t1 - t0) / 1e9);
    }
    double med = median(gops);
    double mn = *std::min_element(gops.begin(), gops.end());
    double mx = *std::max_element(gops.begin(), gops.end());
    printf("  INT8 compute Variant G (grouped horizontal): %7.2f GOP/s  (min: %5.2f, max: %5.2f)\n", med, mn, mx);
    return med;
}

static double int8_compute_variant_l(int reps) {
    const int K = 896, ROWS = 16;
    const long REPEAT = 200000;
    const int nthreads = omp_get_max_threads();
    volatile int32_t sink = 0;
    std::vector<double> gops;
    for (int r = 0; r < reps; r++) {
        double t0 = now_sec();
        #pragma omp parallel
        {
            std::vector<int8_t> w(ROWS * K), a(K);
            for (int i = 0; i < ROWS * K; i++) w[i] = (int8_t)((i * 7 + 3) % 251 - 125);
            for (int i = 0; i < K; i++)        a[i] = (int8_t)((i * 11 + 5) % 251 - 125);
            int32_t total = 0;
            for (long rep = 0; rep < REPEAT; rep++) {
                int32_t round_sum = 0;
                for (int row = 0; row < ROWS; row++) {
                    float lane[16] = {0};
                    for (int g = 0; g < K / 64; g++) {
                        int32_t part[16] = {0};
                        const int8_t* pa = a.data() + g * 64;
                        const int8_t* pw = w.data() + (size_t)row * K + g * 64;
                        for (int k = 0; k < 64; k += 16)
                            for (int l = 0; l < 16; l++) part[l] += (int32_t)pa[k + l] * (int32_t)pw[k + l];
                        for (int l = 0; l < 16; l++) lane[l] += (float)part[l] * 0.01f;
                    }
                    float acc = 0.0f;
                    for (int l = 0; l < 16; l++) acc += lane[l];
                    round_sum += (int32_t)acc;
                }
                a[0] = (int8_t)(round_sum & 0x3f);
                total += round_sum;
            }
            #pragma omp critical
            sink += total;
        }
        double t1 = now_sec();
        gops.push_back((double)nthreads * REPEAT * ROWS * K * 2.0 / (t1 - t0) / 1e9);
    }
    double med = median(gops);
    double mn = *std::min_element(gops.begin(), gops.end());
    double mx = *std::max_element(gops.begin(), gops.end());
    printf("  INT8 compute Variant L (lane-wise accum):   %7.2f GOP/s  (min: %5.2f, max: %5.2f)\n", med, mn, mx);
    return med;
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

static void print_roofline_summary(double peak_bw_gbs, double peak_int8_gops, double peak_fp32_gflops) {
    printf("=== Roofline Summary ===\n");
    printf("  Peak bandwidth:  %.2f GB/s\n", peak_bw_gbs);
    printf("  Peak INT8:       %.2f GOP/s\n", peak_int8_gops);
    printf("  Peak FP32:       %.2f GFLOP/s\n", peak_fp32_gflops);
    printf("  INT8 ridge:      %.2f OP/byte\n", peak_int8_gops / peak_bw_gbs);
    printf("  FP32 ridge:      %.2f FLOP/byte\n", peak_fp32_gflops / peak_bw_gbs);
    printf("\n");

    printf("  --- Operational Intensity for decode ---\n");

    auto classify = [&](const char* name, int M, int elem_bytes, double peak_ops) {
        double oi = 2.0 * M / elem_bytes;                  // ops per weight byte
        double ridge = peak_ops / peak_bw_gbs;
        double attainable = std::min(peak_ops, peak_bw_gbs * oi);
        printf("  %-26s OI %5.1f  ridge %6.1f  attainable %7.1f GOP/s  %s\n",
               name, oi, ridge, attainable, oi < ridge ? "MEMORY-bound" : "COMPUTE-bound");
    };
    classify("INT8 weights, batch 1", 1, 1, peak_int8_gops);
    classify("INT8 weights, batch 8", 8, 1, peak_int8_gops);
    classify("FP32 weights, batch 1", 1, 4, peak_fp32_gflops);
    classify("FP32 weights, batch 8", 8, 4, peak_fp32_gflops);

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
    double bw_triad = stream_benchmark(stream_n, 20);

    // Read-only bandwidth benchmark
    printf("=== Read-Only Bandwidth ===\n");
    double bw_read = read_bandwidth_benchmark(64 * 1024 * 1024, 20);
    printf("\n");

    double peak_bw = std::max(bw_read, bw_triad);

    // 2. FP32 peak (vector FMA)
    printf("=== Peak FP32 Throughput ===\n");
    double fp32 = fp32_vector_benchmark(10);
    printf("\n");

    // 3. INT8 compute (in-cache)
    printf("=== Peak INT8 Throughput ===\n");
    double int8 = int8_compute_benchmark(5);
    int8_compute_variant_g(5);
    int8_compute_variant_l(5);
    printf("\n");

    // 4. LM head effective bandwidth
    lm_head_bandwidth_benchmark(20);

    // 5. Summary
    print_roofline_summary(peak_bw, int8, fp32);

    printf("CSV,mem_best_gbs,%.2f\nCSV,peak_int8_gops,%.2f\nCSV,peak_fp32_gflops,%.2f\n", peak_bw, int8, fp32);

    return 0;
}
