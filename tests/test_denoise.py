import unittest

from diffusion_policy_accelerated import config
from diffusion_policy_accelerated.diffusion_policy.model import generate_diffusion_constants
import diffusion_policy_accelerated.denoise as denoise

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor


class TestDenoiseFunction(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.generator = torch.Generator().manual_seed(config.SEED)
        self.generator_state = self.generator.get_state()
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=100,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
        self.noisy_action = torch.randn((2, 16)).to('cuda')
        self.diffusion_constants = generate_diffusion_constants()
    
    def reset_generator(self):
        """Make sure every time rng is used, noise is identical to initial state."""
        self.generator.set_state(self.generator_state)
        return self.generator

    def test_denoise_similarity(self):
        """Test if custom denoise output is similar to Hugging Face's scheduler output."""
        for k in self.noise_scheduler.timesteps:
            with self.subTest(timestep=k):
                model_output = torch.randn((2, 16)).to('cuda')
                hf_output = self.noise_scheduler.step(
                    model_output=model_output,
                    timestep=k,
                    sample=self.noisy_action,
                    generator=self.reset_generator()
                ).prev_sample
                diffusion_noise = randn_tensor(
                    model_output.shape, generator=self.reset_generator(), device=config.DEVICE, dtype=model_output.dtype
                )
                custom_output = denoise.denoise(model_output, self.noisy_action, self.diffusion_constants, k, diffusion_noise)
                absolute_diff = torch.abs(custom_output - hf_output)
                max_diff = torch.max(absolute_diff).item()
                self.assertLess(max_diff, 0.005, f"Custom output and Hugging Face output do not match at timestep {k}, largest absolute diff is {max_diff}")

if __name__ == '__main__':
    unittest.main()

