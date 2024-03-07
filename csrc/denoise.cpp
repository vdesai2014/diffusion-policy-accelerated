#include <torch/extension.h>

torch::Tensor denoise_cuda(
    torch::Tensor& model_output,
    torch::Tensor& sample,
    torch::Tensor& diffusion_constants,
    torch::Tensor& timestep,
    torch::Tensor& diffusion_noise
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("denoise", &denoise_cuda, "Denoise wrapper function",
          py::arg("model_output"), py::arg("sample"), py::arg("diffusion_constants"), py::arg("timestep"), py::arg("diffusion_noise"));
}
