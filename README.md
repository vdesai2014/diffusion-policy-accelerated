# Diffusion Policy Accelerated

[![PyPI version](https://badge.fury.io/py/diffusion-policy-accelerated.svg)](https://badge.fury.io/py/diffusion-policy-accelerated)

Diffusion Policy Accelerated is a library that showcases the use of custom CUDA extensions and CUDA graphs to accelerate the inference of Diffusion Policy. It provides optimized implementations of convolutional operations and denoising functions to improve the performance of Diffusion Policy models.

## Features

- Custom CUDA extensions for accelerated convolutional operations and denoising functions
- Integration with CUDA graphs for efficient inference
- Easy-to-use API for evaluating Diffusion Policy models
- Support for both inference evaluation and policy evaluation modes

## Installation

To install Diffusion Policy Accelerated, make sure you have PyTorch installed. You can install PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/).

Once PyTorch is installed, you can install Diffusion Policy Accelerated using pip:

pip install diffusion-policy-accelerated

## Usage

Diffusion Policy Accelerated provides a command-line interface for running evaluations with or without inference acceleration.

To run an evaluation, use the diffusion-policy-accelerated command followed by the desired evaluation mode and the number of evaluations to run:

diffusion-policy-accelerated --mode <evaluation_mode> --evals <num_evaluations>

- <evaluation_mode>: Choose between inference-eval for UNet Inference Evaluation or policy-eval for Diffusion Policy Evaluation. Default is inference-eval.
- <num_evaluations>: Specify the number of evaluations to run. Defaults are 100 for inference-eval and 5 for policy-eval.

Example:
'''
diffusion-policy-accelerated --mode policy-eval --evals 10
'''

## Evaluation Modes

Diffusion Policy Accelerated supports two evaluation modes:

1. **Inference Evaluation**: This mode evaluates the performance of the UNet model used in Diffusion Policy. It runs the specified number of evaluations in both PyTorch eager mode and accelerated mode using custom CUDA extensions.

2. **Policy Evaluation**: This mode evaluates the performance of the complete Diffusion Policy model. It runs the specified number of evaluations in both PyTorch eager mode and accelerated mode using custom CUDA extensions and CUDA graphs.

## Contributing

Contributions to Diffusion Policy Accelerated are welcome! If you find any issues or have suggestions for improvements, please open an issue on the [GitHub repository](https://github.com/your-repo-url).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The Diffusion Policy model and the original implementation.
- The PyTorch team for providing the powerful deep learning framework.
- The CUDA team for their excellent work on GPU acceleration.
