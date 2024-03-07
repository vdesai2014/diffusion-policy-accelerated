# Diffusion Policy Accelerated

[![PyPI version](https://badge.fury.io/py/diffusion-policy-accelerated.svg)](https://badge.fury.io/py/diffusion-policy-accelerated)

Diffusion Policy Accelerated is a library that showcases the use of custom CUDA extensions and CUDA graphs to accelerate the inference of DiffusionPolicy-C. It's primary purpose is to serve as a pedagogical tool for those interested in writing custom GPU kernels to improve model inference performance. Refer to [this](https://www.vrushankdes.ai/diffusion-inference-optimization) blog post series for more info. 

## Installation

To install Diffusion Policy Accelerated, make sure you have PyTorch installed. You can install PyTorch by following the instructions on the [official PyTorch website](https://pytorch.org/).

Once PyTorch is installed, you can install Diffusion Policy Accelerated using pip:

pip install diffusion-policy-accelerated

You may have issues running evals in policy-mode if the weights fail to download using gdown. In these cases you can manually download the [weights](https://drive.google.com/uc?id=1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t) and place them in the 'diffusion_policy' folder, relative to the main library directory. You can find the main library directory by running 'which diffusion-policy-accelerated' after installation (on Linux systems). 

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

1. **Inference Evaluation**: This mode evaluates the performance of the Convolutional UNet model used in Diffusion Policy. It runs the specified number of forward-passes in both PyTorch eager mode and accelerated mode using custom CUDA extensions.

2. **Policy Evaluation**: This mode evaluates the performance of the complete DiffusionPolicy-C. It runs the specified number of evaluations in both modes and shows success rates for each episode. 

## Contributing

Contributions to Diffusion Policy Accelerated are welcome! If you find any issues or have suggestions for improvements, please open an issue on the [GitHub repository](https://github.com/your-repo-url).


## Acknowledgments

- The original [Diffusion Policy work](https://github.com/real-stanford/diffusion_policy) which was really well-written and served as a great base for learning purposes!
