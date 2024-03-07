import unittest
import torch
import torch.nn as nn

from diffusion_policy_accelerated import config
from diffusion_policy_accelerated.diffusion_policy.model import Conv1dBlock

class TestConv1dGNMBlock(unittest.TestCase):
    def test_conv1d_block(self):
        for out_channels, in_channels, length in config.TENSOR_SHAPES:
            with self.subTest(out_channels=out_channels, in_channels=in_channels, length=length):
                block = Conv1dBlock(in_channels, out_channels, kernel_size=5, n_groups=8).to(config.DEVICE)

                x = torch.randn(1, in_channels, length, device=config.DEVICE)

                config.inference_mode = config.InferenceMode.NORMAL
                normal_output = block(x)

                config.inference_mode = config.InferenceMode.ACCELERATED
                accelerated_output = block(x)

                self.assertTrue(torch.allclose(normal_output, accelerated_output, atol=5e-3),
                                "Outputs are not close enough between default and custom kernels.")

if __name__ == '__main__':
    unittest.main()
