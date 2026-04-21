#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Single-warp flash attention kernel.
// Fixes the v1 bugs: correct thread-dimension mapping, no wasted threads,
// warp shuffle for dot-product reduction, scalar O accumulator per thread.
//
// Grid:  (N, H, B)  — one warp per query row
// Block: 32 threads  — one warp, no cross-warp sync needed
// Each thread owns D/32 output dimensions throughout the kernel.
// Supports D up to 128 (must be a multiple of 32).

#define WARP_SIZE 32
#define TILE_SIZE 32

// Butterfly warp reduction. After this call every thread in the warp
// holds the total sum — no shared memory or __syncthreads needed.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void flash_attention_v2_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D
) {
    int b   = blockIdx.z;
    int h   = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;  // 0..31

    int dims_per_thread = D / WARP_SIZE;  // e.g. 64/32 = 2

    extern __shared__ float smem[];
    float* Q_s = smem;                     // D floats
    float* K_s = Q_s + D;                  // TILE_SIZE * D floats
    float* V_s = K_s + TILE_SIZE * D;      // TILE_SIZE * D floats

    int qo_base = ((b * H + h) * N + row) * D;

    // Load this row's Q vector into shared memory (coalesced)
    for (int i = tid; i < D; i += WARP_SIZE)
        Q_s[i] = Q[qo_base + i];
    __syncwarp();

    // Per-thread output accumulators — one slot per owned dimension
    float o_local[4] = {0.f, 0.f, 0.f, 0.f};  // supports D up to 128

    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Tile over K/V columns
    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
        int tile_size = min(TILE_SIZE, N - tile_start);
        int kv_base   = ((b * H + h) * N + tile_start) * D;

        // Cooperative coalesced load of K and V tiles
        for (int i = tid; i < tile_size * D; i += WARP_SIZE) {
            K_s[i] = K[kv_base + i];
            V_s[i] = V[kv_base + i];
        }
        __syncwarp();

        // Process each key/value in the tile
        for (int col = 0; col < tile_size; col++) {

            // --- dot(Q[row], K[tile_start+col]) / sqrt(D) ---
            float partial = 0.0f;
            #pragma unroll
            for (int i = 0; i < dims_per_thread; i++) {
                int d = tid + i * WARP_SIZE;
                partial += Q_s[d] * K_s[col * D + d];
            }
            // Warp butterfly — all 32 threads get the full dot product
            float score = warp_reduce_sum(partial) / sqrtf((float)D);

            // --- online softmax + accumulate V ---
            float new_max = fmaxf(row_max, score);
            float exp_old = expf(row_max - new_max);
            float exp_new = expf(score  - new_max);

            // Rescale running accumulator and denominator
            #pragma unroll
            for (int i = 0; i < dims_per_thread; i++)
                o_local[i] *= exp_old;
            row_sum = row_sum * exp_old + exp_new;

            // Accumulate weighted V
            #pragma unroll
            for (int i = 0; i < dims_per_thread; i++) {
                int d = tid + i * WARP_SIZE;
                o_local[i] += exp_new * V_s[col * D + d];
            }

            row_max = new_max;
        }
        __syncwarp();
    }

    // Write final output: O[b][h][row][d] = o_local[d] / row_sum
    for (int i = 0; i < dims_per_thread; i++) {
        int d = tid + i * WARP_SIZE;
        O[qo_base + d] = o_local[i] / row_sum;
    }
}

torch::Tensor flash_attention_v2_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    TORCH_CHECK(D <= 128,            "flash_v2 supports D <= 128");
    TORCH_CHECK(D % WARP_SIZE == 0,  "D must be a multiple of 32");

    auto O = torch::zeros({B, H, N, D}, Q.options());

    dim3 grid(N, H, B);
    size_t smem_bytes = (D + 2 * TILE_SIZE * D) * sizeof(float);

    flash_attention_v2_kernel<<<grid, WARP_SIZE, smem_bytes>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, N, D
    );

    return O;
}
