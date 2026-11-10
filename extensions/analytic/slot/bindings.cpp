#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> analyze_slots_cuda(
    torch::Tensor colors, torch::Tensor geometry, torch::Tensor binning,
    torch::Tensor image_state, torch::Tensor raw_rgb, torch::Tensor gt,
    torch::Tensor image_gradient, torch::Tensor background, int64_t rendered);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("analyze", &analyze_slots_cuda,
          "Read-only fixed-contributor opacity-direction moments (CUDA)");
}
