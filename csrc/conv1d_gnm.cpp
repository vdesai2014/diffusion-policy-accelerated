#include <ATen/ATen.h>
#include <torch/extension.h>
#include <vector>

torch::Tensor conv1d_gnm_cuda(
    torch::Tensor& input,
    torch::Tensor& conv1d_weight_tensor,
    torch::Tensor& conv1d_bias_tensor,
    torch::Tensor& gnm_weight_tensor,
    torch::Tensor& gnm_bias_tensor,
    int output_channels,
    int padding,
    int kernel_size,
    const int channels_per_thread
);

torch::Tensor conv1d_gnm_wrapper(
    torch::Tensor& input,
    torch::Tensor& conv1d_weight_tensor,
    torch::Tensor& conv1d_bias_tensor,
    torch::Tensor& gnm_weight_tensor,
    torch::Tensor& gnm_bias_tensor,
    int output_channels,
    int padding,
    int kernel_size,
    const int channels_per_thread)
{
    at::DeviceGuard guard(input.device());
    return conv1d_gnm_cuda(input, conv1d_weight_tensor, conv1d_bias_tensor, gnm_weight_tensor, gnm_bias_tensor, output_channels, padding, kernel_size, channels_per_thread);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("conv1d_gnm", &conv1d_gnm_wrapper, "1D convolution + gnm wrapper function",
          py::arg("input"), py::arg("conv1d_weight_tensor"), py::arg("conv1d_bias_tensor"), py::arg("gnm_weight_tensor"), py::arg("gnm_bias_tensor"), py::arg("output_channels"), py::arg("padding"), py::arg("kernel_size"), py::arg("channels_per_thread"));
}
