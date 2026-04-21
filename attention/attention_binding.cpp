#include <torch/extension.h>

torch::Tensor naive_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor optimized_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor flash_attention_v2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
torch::Tensor flash_attention_v3_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_forward",     &naive_attention_forward,     "Naive Attention (Stage 1)");
    m.def("optimized_forward", &optimized_attention_forward, "Optimized Attention (Stage 2 - broken)");
    m.def("flash_forward",     &flash_attention_forward,     "Flash Attention v1 (Stage 3 - broken)");
    m.def("flash_v2_forward",  &flash_attention_v2_forward,  "Flash Attention v2 (Stage 4)");
    m.def("flash_v3_forward",  &flash_attention_v3_forward,  "Flash Attention v3 (Stage 5)");
}
