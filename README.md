# Velox Inference Engine

A C++ inference engine for running Qwen2.5-style transformer models on CPU. It loads models directly from [safetensors](https://github.com/huggingface/safetensors) files, quantizes weights to INT8, and runs batched autoregressive generation with a growable KV cache - no PyTorch or Python runtime required at inference time.

## Techniques

- **Memory-mapped model loading** - [`safetensors.cpp`](src/safetensors.cpp) maps the model file directly into memory with [`mmap`](https://man7.org/linux/man-pages/man2/mmap.2.html) instead of reading it into a buffer. Tensors are read straight from the mapped region, so the OS handles paging and the engine avoids a full-file copy on startup.
- **Row-wise INT8 quantization (W8A8)** - Weights are quantized once at load time in [`ops::quantize_rowwise`](src/ops.cpp), with one scale factor per output row. Activations are quantized dynamically on every forward pass in [`ops::matmul_w8a8`](src/ops.cpp) before the integer matmul runs, then rescaled back to float. This cuts memory bandwidth roughly 4x compared to FP32 weights.
- **Grouped-query attention** - [`model.hpp`](include/model.hpp) configures fewer key/value heads (`num_key_value_heads`) than query heads (`num_attention_heads`), so multiple query heads share the same K/V projections. This shrinks the KV cache without changing attention quality much. See the [original GQA paper](https://arxiv.org/abs/2305.13245) for background.
- **Rotary position embeddings (RoPE)** - [`ops::rope`](src/ops.cpp) rotates query/key vectors by an angle derived from token position instead of adding a position vector. It encodes relative position directly into the attention dot product.
- **Amortized KV cache growth** - [`KVCache::ensure_capacity`](include/model.hpp) doubles the cache buffer whenever it runs out of room, the same growth strategy used internally by [`std::vector`](https://en.cppreference.com/w/cpp/container/vector). This keeps reallocation cost to `O(1)` amortized per token instead of resizing on every step.
- **Cache-blocked matrix multiplication** - the fallback FP32 [`ops::matmul`](src/ops.cpp) tiles the computation into 32x32 blocks rather than looping naively, keeping working data inside CPU cache lines longer.
- **Conditional OpenMP parallelization** - attention loops use [`#pragma omp parallel for collapse(2)`](https://www.openmp.org/spec-html/5.0/openmpsu41.html), but only when `batch_size * num_heads` is large enough to justify thread spin-up. Small single-sequence workloads run single-threaded to avoid overhead.
- **BF16 to FP32 conversion via bit manipulation** - [`SafetensorsFile::bf16_to_float`](src/safetensors.cpp) reconstructs a float from a bfloat16 value by shifting its bits into the upper 16 bits of a 32-bit float, since BF16 is just a truncated FP32.
- **Byte-level BPE tokenizer without regex** - [`tokenizer.cpp`](src/tokenizer.cpp) reimplements GPT-2's byte-level pre-tokenizer using manual character classification instead of a regex engine, matching Hugging Face's `transformers` tokenizer output.
- **Non-owning tensor views** - [`Tensor::view`](include/tensor.hpp) and [`Tensor::row`](include/tensor.hpp) reinterpret existing memory with a new shape instead of copying data, similar to how NumPy views work.

## Notable Libraries & Technologies

- [OpenMP](https://www.openmp.org/) - compiler-level multithreading used for parallel loops across attention heads, quantization, and normalization.
- [safetensors](https://github.com/huggingface/safetensors) - the model file format this engine reads directly, designed for safe, fast, memory-mappable tensor storage.
- [nlohmann/json](https://github.com/nlohmann/json) (`json.hpp`) - parses `config.json` and `tokenizer.json` at load time.
- [Qwen2.5](https://huggingface.co/Qwen) - the model architecture this engine implements (grouped-query attention, RoPE, RMSNorm, SwiGLU MLP).
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - used only in [`scripts/verify.py`](scripts/verify.py) to generate reference logits for correctness checks against the C++ output.
- [CMake](https://cmake.org/) 3.16+ - build system, configured for `-O3 -march=native` release builds.

## Project Structure

```
.
├── CMakeLists.txt
├── README.md
├── include/
├── scripts/
└── src/
```

- [`include/`](include/) - public headers defining the model, tensor, tokenizer, sampler, and safetensors interfaces.
- [`src/`](src/) - implementation files, including the main inference loop and CLI entry point in [`main.cpp`](src/main.cpp).
- [`scripts/`](scripts/) - Python tooling for cross-checking output against Hugging Face `transformers`.

## Benchmarks

Measured on a Qwen2.5-0.5B model, 15 repetitions per batch size, median ± standard deviation:

| Batch Size | TTFT (ms) | Per-Seq Speed (tok/s) | Aggregate Throughput (tok/s) |
|------------|-----------|------------------------|-------------------------------|
| 1 | 236 ± 151 | 3.34 ± 0.25 | 3.34 ± 0.25 |
| 2 | 440 ± 225 | 2.06 ± 0.09 | 4.13 ± 0.19 |
| 4 | 849 ± 143 | 1.44 ± 0.08 | 5.74 ± 0.32 |
| 8 | 1855 ± 207 | 0.74 ± 0.00 | 5.94 ± 0.03 |

Time to First Token (TTFT) is a single point-in-time measurement, so it's noisy on virtualized environments like WSL2 - a stolen CPU core can spike one run without affecting the trend. Aggregate throughput averages over many decode steps, so it stays stable even with fewer repetitions. Run 15+ repetitions if you need a reliable TTFT median.
