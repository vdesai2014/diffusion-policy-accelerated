import sys
from setuptools import setup, find_packages

BRIGHT_ORANGE = '\033[93m'
RESET_COLOR = '\033[0m'

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as e:
    error_message = "Error: Torch must be installed before installing this package.\n"
    colored_error_message = f"{BRIGHT_ORANGE}{error_message}{RESET_COLOR}"
    sys.stderr.write(colored_error_message)
    sys.exit(1)

setup(
    name='diffusion_policy_accelerated',
    version='1.4',
    author="Vrushank Desai",
    author_email="vrushank@vrushankdes.ai",
    description=("A library to showcase the use of custom CUDA extensions & CUDA graphs to accelerate the inference of Diffusion Policy."),
    packages=find_packages(include=['diffusion_policy_accelerated', 'diffusion_policy_accelerated.*']),
    ext_modules=[
        CUDAExtension(
            name='diffusion_policy_accelerated.conv1d_gnm',
            sources=['csrc/conv1d_gnm.cpp', 'csrc/conv1d_gnm_kernel.cu'],
        ),
        CUDAExtension(
            name='diffusion_policy_accelerated.denoise',
            sources=['csrc/denoise.cpp', 'csrc/denoise_kernel.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.12.0', 
        'tqdm>=4.66.1',
        'diffusers>=0.26.3',
        'numpy>=1.21.2', 
        'gym>=0.26.2',
        'pygame>=2.1.0', 
        'pymunk>=6.2.0',  
        'shapely>=1.8.0',
        'opencv-python>=4.5.5.62',
        'scikit-image>=0.18.3', 
        'scikit-video>=1.1.11',
        'IPython>=7.30.0', 
        'torchvision>=0.13.0',  
        'gdown>=4.4.0', 
        'zarr>=2.10.0' 
    ],
    entry_points={
        'console_scripts': [
            'diffusion-policy-accelerated=diffusion_policy_accelerated.main:main',
        ],
    }
)
