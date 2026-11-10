#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> analyze_anchor_groups_cuda(torch::Tensor colors,torch::Tensor geometry,
    torch::Tensor binning,torch::Tensor image_state,torch::Tensor rgb,torch::Tensor background,
    torch::Tensor parents,torch::Tensor selected,int64_t rendered,int64_t anchors,int64_t capacity);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("analyze",&analyze_anchor_groups_cuda,"Read-only grouped local opacity-direction energy; exact bounded fallback (CUDA)");
}
