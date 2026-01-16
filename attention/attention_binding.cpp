#include <torch/extension.h>

// Forward declarations — REQUIRED
torch::Tensor naive_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

torch::Tensor optimized_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

torch::Tensor flash_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_forward", &naive_attention_forward, "Naive Attention");
    m.def("optimized_forward", &optimized_attention_forward, "Optimized Attention");
    m.def("flash_forward", &flash_attention_forward, "Flash Attention");
}
