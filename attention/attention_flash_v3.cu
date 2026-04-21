#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Flash Attention v3: multi-row blocks.
//
// Key insight: all query rows in the same (b, h) share the same K and V.
// By packing WARPS_PER_BLOCK rows into one block, K/V tiles are loaded
// once and reused across all warps — cutting global memory bandwidth
// by WARPS_PER_BLOCK× for the K/V reads.
//
// Grid:  (ceil(N / WARPS_PER_BLOCK), H, B)
// Block: WARPS_PER_BLOCK * 32 threads
// Each warp handles one query row, same as v2.

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 4
#define BLOCK_DIM (WARP_SIZE * WARPS_PER_BLOCK)  // 128
#define TILE_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ __launch_bounds__(BLOCK_DIM)
void flash_attention_v3_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D
) {
    int b        = blockIdx.z;
    int h        = blockIdx.y;
    int row_base = blockIdx.x * WARPS_PER_BLOCK;

    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;    // 0..WARPS_PER_BLOCK-1
    int lane_id = tid % WARP_SIZE;    // 0..31

    int row       = row_base + warp_id;
    bool row_valid = (row < N);

    int dims_per_thread = D / WARP_SIZE;

    // Shared memory layout:
    //   Q_s : WARPS_PER_BLOCK * D  (each warp has its own Q row)
    //   K_s : TILE_SIZE * D        (shared across all warps)
    //   V_s : TILE_SIZE * D        (shared across all warps)
    extern __shared__ float smem[];
    float* Q_s = smem;
    float* K_s = Q_s + WARPS_PER_BLOCK * D;
    float* V_s = K_s + TILE_SIZE * D;

    float* my_Q = Q_s + warp_id * D;

    int qo_base = ((b * H + h) * N + row) * D;
    int bh_base = (b * H + h) * N * D;

    // Each warp loads its own Q row
    if (row_valid) {
        for (int i = lane_id; i < D; i += WARP_SIZE)
            my_Q[i] = Q[qo_base + i];
    }
    // No sync needed yet — warps only read their own Q_s section.
    // The first __syncthreads() below covers K/V loading.

    float o_local[4] = {0.f, 0.f, 0.f, 0.f};
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
        int tile_size = min(TILE_SIZE, N - tile_start);
        int kv_offset = tile_start * D;

        // ALL 128 threads cooperatively load K and V (coalesced)
        for (int i = tid; i < tile_size * D; i += BLOCK_DIM) {
            K_s[i] = K[bh_base + kv_offset + i];
            V_s[i] = V[bh_base + kv_offset + i];
        }
        __syncthreads();

        if (row_valid) {
            for (int col = 0; col < tile_size; col++) {

                // dot(Q[row], K[tile_start + col]) / sqrt(D)
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < dims_per_thread; i++) {
                    int d = lane_id + i * WARP_SIZE;
                    partial += my_Q[d] * K_s[col * D + d];
                }
                float score = warp_reduce_sum(partial) / sqrtf((float)D);

                // Online softmax + accumulate V
                float new_max = fmaxf(row_max, score);
                float exp_old = expf(row_max - new_max);
                float exp_new = expf(score  - new_max);

                #pragma unroll
                for (int i = 0; i < dims_per_thread; i++)
                    o_local[i] *= exp_old;
                row_sum = row_sum * exp_old + exp_new;

                #pragma unroll
                for (int i = 0; i < dims_per_thread; i++) {
                    int d = lane_id + i * WARP_SIZE;
                    o_local[i] += exp_new * V_s[col * D + d];
                }

                row_max = new_max;
            }
        }
        __syncthreads();  // protect K_s/V_s before next tile load
    }

    // Write output
    if (row_valid) {
        for (int i = 0; i < dims_per_thread; i++) {
            int d = lane_id + i * WARP_SIZE;
            O[qo_base + d] = o_local[i] / row_sum;
        }
    }
}

torch::Tensor flash_attention_v3_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    TORCH_CHECK(D <= 128,            "flash_v3 supports D <= 128");
    TORCH_CHECK(D % WARP_SIZE == 0,  "D must be a multiple of 32");

    auto O = torch::zeros({B, H, N, D}, Q.options());

    int grid_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(grid_x, H, B);
    size_t smem_bytes = (WARPS_PER_BLOCK * D + 2 * TILE_SIZE * D) * sizeof(float);

    flash_attention_v3_kernel<<<grid, BLOCK_DIM, smem_bytes>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, N, D
    );

    return O;
}
