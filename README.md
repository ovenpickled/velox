# Velox Inference Engine

A high-performance C++ inference engine for large language models.

## Architecture

- **KV Cache**: Implements dynamic capacity scaling with $O(1)$ amortized memory growth. The cache size doubles dynamically when capacity is exceeded to minimize heap allocations.
- **Pre-tokenizer**: Custom C++ implementation of the GPT-2 byte-level pre-tokenizer. It accurately groups whitespace and punctuation without needing heavy regex libraries, fully matching the behavior of the `transformers` library.
- **Parallelization**: Critical loops (Attention, RMSNorm, Bias) are parallelized using OpenMP `collapse(2)`. The parallelization is conditionally gated to avoid thread spin-up overhead on small workloads (like single-sequence generation).

## Performance Notes

When benchmarking on virtualized environments (like WSL2), you may observe a structural difference in noise between **Time to First Token (TTFT)** and **Aggregate Throughput**:

- **TTFT Variance**: TTFT is a single point-in-time measurement per iteration. Because prefill happens extremely fast, there is no internal averaging to smooth out OS jitter (e.g., a background process momentarily stealing a core). Therefore, TTFT naturally exhibits a much higher relative variance (often >25%), and small batch sizes can occasionally spike higher than larger batches if an outlier event occurs.
- **Throughput Variance**: Aggregate throughput is computed by averaging over multiple decode steps (e.g., 128 tokens) within a single iteration. This internal averaging dilutes one-off jitter events, resulting in a much tighter relative variance (often <5%) and a very clean, monotonic scaling curve as batch sizes increase.

To account for this structural noise, benchmarks should be run with a high number of repetitions (e.g., 15+) to extract a stable median for TTFT, while throughput will remain stable even with fewer repetitions.