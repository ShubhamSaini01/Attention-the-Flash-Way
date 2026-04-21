# Attention the Flash Way

A ground-up CUDA implementation of scaled dot-product attention, evolved through five stages from a naive three-kernel approach to a warp-optimized Flash Attention kernel. Each stage is benchmarked against PyTorch's `scaled_dot_product_attention` for both latency and correctness.

## Benchmark results

Config: `B=1, H=8, N=512, D=64` (100 runs, 10 warmup)

| Kernel | Median (ms) | vs PyTorch SDPA | Max error |
|---|---|---|---|
| PyTorch SDPA | 0.095 | 1.0× | — |
| **Flash v3 (4-warp)** | **0.766** | **8.1×** | 1e-6 |
| Naive | 1.344 | 14.1× | 1e-6 |
| Flash v2 (1-warp) | 2.579 | 27.1× | 1e-6 |
| Optimized | 20.447 | 215× | 1e-6 |
| Flash v1 | 104.396 | 1099× | 1e-6 |

## Evolution

### Stage 1 — Naive (`attention_naive.cu`)

Three kernels chained together:
1. `qk_kernel` — computes `S = QK^T / sqrt(D)`, one `(row, col)` element per thread
2. `softmax_kernel` — serial per-row softmax in global memory
3. `sv_kernel` — computes `O = S @ V`

Materializes the full N×N score matrix in global memory. Embarrassingly parallel — no synchronization overhead, which is why it outperforms the broken "optimized" versions below.

### Stage 2 — Optimized (`attention_optimized.cu`) — broken

Intended to improve over naive with shared memory for K and parallel reduction for softmax. **Actually 15× slower than naive** due to a parallelism bug: all 256 threads compute the same dot product but only thread 0 writes the result. 255 out of 256 threads do completely wasted work, plus 1024 `__syncthreads()` barriers per block from serializing N columns.

### Stage 3 — Flash v1 (`attention_flash.cu`) — broken

Correct Flash Attention math (online softmax, tiled K/V, register accumulator) but **75× slower than naive** due to two compounding bugs:
- **Stack spill**: `float O_local[64]` + `float scores[32]` overflow registers into **384 bytes of stack spill per thread** (backed by slow L2/DRAM local memory). Confirmed via `cuobjdump`: v1 uses 31 REG + 384 STACK vs v2's 32 REG + 0 STACK.
- **Wasted threads**: all 256 threads compute identical dot products but only 64 write output (4× waste)

### Stage 4 — Flash v2 (`attention_flash_v2.cu`)

Fixes v1 by using **one warp (32 threads) per query row**:
- Each thread owns `D/32 = 2` output dimensions throughout the kernel
- Dot products via **warp butterfly shuffle** (`__shfl_xor_sync`) — all 32 threads get the result with zero shared memory
- `o_local` is a scalar per thread, not a 64-element array — eliminates the 384-byte stack spill entirely (32 REG, 0 STACK per `cuobjdump`)
- Online softmax computed per-element — no scores array needed

**Result: 40× faster than v1** (2.58ms vs 104ms). But 2× slower than naive because only 32 threads per block limits occupancy.

### Stage 5 — Flash v3 (`attention_flash_v3.cu`)

Packs **4 warps into one block**, each handling a different query row. The critical insight: all rows in the same `(batch, head)` share the same K and V, so K/V tiles are **loaded once and reused across 4 warps**.

Benefits over v2:
- **4× less global memory bandwidth** for K/V reads (amortized across 4 rows)
- **4× better occupancy** (128 threads/block vs 32)
- **4× faster tile loads** (128 threads cooperating)

**Result: 3.4× faster than v2, 1.75× faster than naive** (0.77ms). The remaining 8× gap to PyTorch SDPA is almost entirely FP16 + Tensor Cores.

## Build and run

Requires PyTorch with CUDA support.

```bash
python setup.py install
python benchmark.py            # Quick comparison at N=512
python benchmark.py --scale    # Scaling test across N=256..16384
```

## Project structure

```
attention/
  attention_naive.cu          # Stage 1: naive three-kernel attention
  attention_optimized.cu      # Stage 2: shared memory (broken parallelism)
  attention_flash.cu          # Stage 3: fused flash attention v1 (broken thread mapping)
  attention_flash_v2.cu       # Stage 4: single-warp flash with warp shuffle
  attention_flash_v3.cu       # Stage 5: multi-row blocks, shared K/V tiles
  attention_binding.cpp       # PyBind11 bindings for all 5 stages
benchmark.py                  # Latency + correctness benchmark vs PyTorch SDPA
setup.py                      # PyTorch C++ extension build
docs/presentation.html        # Full optimization journey with NVIDIA doc citations
```

## Roadmap

This is an ongoing exploration of GPU architecture, from first principles to hardware-specific exploitation:

- [x] FP32 Flash Attention with online softmax
- [x] Warp shuffle reductions, multi-row K/V tile sharing
- [ ] FP16 path with Tensor Cores (`wmma` / `mma.sync`)
- [ ] INT8 QK dot product via `dp4a` + FP32 softmax
- [ ] Blackwell-specific optimizations (5th-gen Tensor Cores, custom MMA instructions)
- [ ] Larger tile sizes, double buffering, software pipelining

## Key concepts

- **Online softmax**: track running `max` and `sum` as tiles are processed so the full row never needs to be in memory at once — the core trick in [Flash Attention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- **Warp shuffle reduction**: `__shfl_xor_sync` butterfly pattern computes a 32-thread sum in 5 steps with no shared memory — all threads receive the result
- **K/V tile sharing**: multiple query rows in the same block amortize the cost of loading K/V from global memory
- **Register vs stack spill tradeoff**: large per-thread arrays (`float arr[64]`) overflow registers into stack (local memory backed by L2/DRAM); mapping one thread to one output dimension eliminates spill entirely — confirmed via `cuobjdump` (v1: 31 REG + 384 STACK → v2: 32 REG + 0 STACK)
