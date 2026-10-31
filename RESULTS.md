# Velox: results summary

## Decode performance (Qwen2.5-0.5B, Ryzen 5 220)
Batch-1 decode went from 9.5 to 38.3 tok/s (4.0×), batch-8 aggregate throughput from 11.0 to 144 tok/s (13×)

## Where decode time went
Originally the FP32 LM head was 81% of a batch-1 step (85 of 105 ms) and 92% of a batch-8 step. It ran on one thread, because the matmul parallelised over the batch rows. Parallelising over the output dimension and quantising the LM head to INT8 removed the bottleneck.
Final profile, batch 8 (4 threads): total 68.9 ms: lm_head 15.8, ffn_proj 43.1, qkv 3.7, o_proj 2.8, attention 2.0, ffn_act 1.05, everything else under 0.4. Parallelising silu/multiply cut ffn_act from 3.25 to 1.05 ms; the other buckets moved about 10% between sessions, which is machine variation.
Final profile, batch 1 (4 threads), average ms per decode step. Percentages are shares of the timed layer buckets:

| Component | ms | share of layers |
|---|---|---|
| ffn_proj (gate, up, down) | 16.28 | 81.3% |
| qkv_proj | 1.34 | 6.7% |
| o_proj | 1.24 | 6.2% |
| attention | 0.64 | 3.2% |
| ffn_act (silu, multiply) | 0.31 | 1.5% |
| rope + bias + KV append | 0.08 | 0.4% |
| input_norm | 0.06 | 0.3% |
| ffn_norm | 0.05 | 0.2% |
| **layers total** | **20.01** | |
| lm_head | 5.3 | |
| **forward total** | **25.3** | |

Both big consumers are close to the memory ceiling. Estimated from byte counts: the MLP streams about 405 MB per step (283 MB INT8 plus 122 MB for the seven protected FP32 matrices), which in 16.3 ms is about 25 GB/s, roughly 96% of the 25.8 GB/s read bandwidth. The INT8 LM head streams 136 MB in 5.3 ms, about 25.7 GB/s, which is at the ceiling. Batch-1 decode is bandwidth-bound, so further gains would need fewer bytes, not faster arithmetic.

## Roofline
Measured: 25.8 GB/s read bandwidth, 271 GOP/s INT8 compute at 4 threads (autovectorised kernel). Batch-1 decode sits at about 87% of the memory roof. Batch-8 decode is at about 41% of its roof, so the batch-8 kernel is the main remaining headroom. See docs/roofline.png.

## Accuracy
Verified against Hugging Face on real weights: with everything in FP32 the engine matches to under 1e-3 mean logit difference. The INT8 error was concentrated at position 0, where "massive activations" (|x| ~ 1000) multiply INT8 weights in layer 2's down projection and layers 3 and 21's MLP. Keeping those 7 matrices in FP32 (detected automatically at load time) cut mean error from 1.2 to about 0.14 and fixed top-1 flips, at -12% batch-1 speed.

## Fixes along the way
Heap overflows past 4096 tokens and on empty prompts, tokenizer pre-tokenization (digits, Unicode, punctuation, special tokens, EOS), loader validation and F16 support, UTF-8 streaming, argument handling.

## Known limitations
See README.