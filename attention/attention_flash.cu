#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256
#define TILE_SIZE 32

__global__ void flash_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* Q_tile = smem;
    float* K_tile = Q_tile + TILE_SIZE * D;
    float* V_tile = K_tile + TILE_SIZE * D;

    float max_val = -INFINITY;
    float sum_val = 0.0f;

    // ⚠ assumes D <= 64 (fine for your benchmarks, but document it)
    float O_local[64];

    for (int d = tid; d < D; d += blockDim.x)
        O_local[d] = 0.0f;

    // Load Q
    for (int d = tid; d < D; d += blockDim.x) {
        int q_idx = ((b * H + h) * N + row) * D + d;
        Q_tile[d] = Q[q_idx];
    }
    __syncthreads();

    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
        int tile_size = min(TILE_SIZE, N - tile_start);

        // Load K
        for (int i = tid; i < tile_size * D; i += blockDim.x) {
            int col = i / D;
            int d   = i % D;
            int k_idx = ((b * H + h) * N + (tile_start + col)) * D + d;
            K_tile[col * D + d] = K[k_idx];
        }
        __syncthreads();

        // Load V
        for (int i = tid; i < tile_size * D; i += blockDim.x) {
            int col = i / D;
            int d   = i % D;
            int v_idx = ((b * H + h) * N + (tile_start + col)) * D + d;
            V_tile[col * D + d] = V[v_idx];
        }
        __syncthreads();

        float scores[TILE_SIZE];
        float tile_max = -INFINITY;

        for (int col = 0; col < tile_size; col++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += Q_tile[d] * K_tile[col * D + d];

            scores[col] = dot / sqrtf((float)D);
            tile_max = fmaxf(tile_max, scores[col]);
        }

        float new_max = fmaxf(max_val, tile_max);
        float exp_old = expf(max_val - new_max);

        for (int d = 0; d < D; d++)
            O_local[d] *= exp_old;

        sum_val *= exp_old;

        float tile_sum = 0.0f;
        for (int col = 0; col < tile_size; col++) {
            scores[col] = expf(scores[col] - new_max);
            tile_sum += scores[col];
            for (int d = 0; d < D; d++)
                O_local[d] += scores[col] * V_tile[col * D + d];
        }

        sum_val += tile_sum;
        max_val = new_max;
        __syncthreads();
    }

    for (int d = tid; d < D; d += blockDim.x) {
        int o_idx = ((b * H + h) * N + row) * D + d;
        O[o_idx] = O_local[d] / sum_val;
    }
}

torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    auto O = torch::zeros({B, H, N, D}, Q.options());

    dim3 grid(N, H, B);
    size_t smem = 3 * TILE_SIZE * D * sizeof(float);

    flash_attention_kernel<<<grid, BLOCK_SIZE, smem>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, N, D
    );

    return O;
}
