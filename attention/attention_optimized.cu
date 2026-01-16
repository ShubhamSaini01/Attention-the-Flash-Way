#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#define BLOCK_SIZE 256

__global__ void qk_kernel_opt(
    const float* Q,
    const float* K,
    float* S,
    int B, int H, int N, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x;

    extern __shared__ float K_shared[];

    for (int col = 0; col < N; col++) {
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            int k_idx = ((b * H + h) * N + col) * D + d;
            K_shared[d] = K[k_idx];
        }
        __syncthreads();

        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            int q_idx = ((b * H + h) * N + row) * D + d;
            sum += Q[q_idx] * K_shared[d];
        }

        if (threadIdx.x == 0) {
            int s_idx = ((b * H + h) * N + row) * N + col;
            S[s_idx] = sum / sqrtf((float)D);
        }
        __syncthreads();
    }
}

__global__ void softmax_kernel_opt(
    float* S,
    int B, int H, int N
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float smem[];

    int offset = ((b * H + h) * N + row) * N;

    float max_val = -INFINITY;
    for (int i = tid; i < N; i += blockDim.x)
        max_val = fmaxf(max_val, S[offset + i]);

    smem[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }

    max_val = smem[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float e = expf(S[offset + i] - max_val);
        S[offset + i] = e;
        sum += e;
    }

    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    sum = smem[0];
    __syncthreads();

    for (int i = tid; i < N; i += blockDim.x)
        S[offset + i] /= sum;
}

__global__ void sv_kernel_opt(
    const float* S,
    const float* V,
    float* O,
    int B, int H, int N, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;

    for (int d = tid; d < D; d += blockDim.x) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            int s_idx = ((b * H + h) * N + row) * N + i;
            int v_idx = ((b * H + h) * N + i) * D + d;
            sum += S[s_idx] * V[v_idx];
        }

        int o_idx = ((b * H + h) * N + row) * D + d;
        O[o_idx] = sum;
    }
}

torch::Tensor optimized_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
) {
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int N = Q.size(2);
    const int D = Q.size(3);

    auto S = torch::zeros({B, H, N, N}, Q.options());
    auto O = torch::zeros({B, H, N, D}, Q.options());

    dim3 grid(N, H, B);

    qk_kernel_opt<<<grid, BLOCK_SIZE, D * sizeof(float)>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), S.data_ptr<float>(),
        B, H, N, D
    );

    softmax_kernel_opt<<<grid, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        S.data_ptr<float>(), B, H, N
    );

    sv_kernel_opt<<<grid, BLOCK_SIZE>>>(
        S.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
        B, H, N, D
    );

    return O;
}
