#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__global__ void qk_kernel(
    const float* Q,
    const float* K,
    float* S,
    int B, int H, int N, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (col >= N) return;

    float sum = 0.0f;
    for (int d = 0; d < D; d++) {
        int q_idx = ((b * H + h) * N + row) * D + d;
        int k_idx = ((b * H + h) * N + col) * D + d;
        sum += Q[q_idx] * K[k_idx];
    }

    int s_idx = ((b * H + h) * N + row) * N + col;
    S[s_idx] = sum / sqrtf((float)D);
}

__global__ void softmax_kernel(float* S, int B, int H, int N) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x;

    int offset = ((b * H + h) * N + row) * N;

    float max_val = -INFINITY;
    for (int i = 0; i < N; i++)
        max_val = fmaxf(max_val, S[offset + i]);

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        S[offset + i] = expf(S[offset + i] - max_val);
        sum += S[offset + i];
    }

    for (int i = 0; i < N; i++)
        S[offset + i] /= sum;
}

__global__ void sv_kernel(
    const float* S,
    const float* V,
    float* O,
    int B, int H, int N, int D
) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (col >= D) return;

    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        int s_idx = ((b * H + h) * N + row) * N + i;
        int v_idx = ((b * H + h) * N + i) * D + col;
        sum += S[s_idx] * V[v_idx];
    }

    int o_idx = ((b * H + h) * N + row) * D + col;
    O[o_idx] = sum;
}

torch::Tensor naive_attention_forward(
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

    dim3 grid_qk(N, H, B);
    dim3 block_qk(N);

    qk_kernel<<<grid_qk, block_qk>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        S.data_ptr<float>(),
        B, H, N, D
    );

    dim3 grid_sm(N, H, B);
    softmax_kernel<<<grid_sm, 1>>>(S.data_ptr<float>(), B, H, N);

    dim3 grid_sv(N, H, B);
    dim3 block_sv(D);

    sv_kernel<<<grid_sv, block_sv>>>(
        S.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B, H, N, D
    );

    return O;
}
