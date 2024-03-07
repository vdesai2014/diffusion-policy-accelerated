import time 
import torch 
from tqdm import tqdm
import argparse

import diffusion_policy_accelerated.config as config 
from diffusion_policy_accelerated.diffusion_policy.eval import ModelManager, MDPEnvironment, run_evaluation
from diffusion_policy_accelerated.diffusion_policy.model import ConditionalUnet1D, load_noise_pred_net_graph
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def unet_inference_eval_eager(n, noise_pred_net):
    """
    Performs inference evaluation in eager mode for a given number of iterations.

    This function evaluates the inference time of the noise prediction network in PyTorch eager mode. It generates random
    noisy actions and observation conditions, then uses the noise prediction network and a Huggingface DDPM scheduler to denoise the actions.
    The function measures and prints the total and average time taken for the specified number of iterations.

    Parameters:
    - n (int): The number of iterations to run the evaluation.
    - noise_pred_net (torch.nn.Module): The noise prediction network to be evaluated.
    """

    with config.inference_mode_context(config.InferenceMode.NORMAL):
        noise_scheduler = DDPMScheduler(num_train_timesteps=100)
        start_normal_time = time.perf_counter()
        for _ in tqdm(range(n), desc='Inference Eval - Pytorch Eager Mode'):
            noisy_action = torch.randn((1, config.PRED_HORIZON, config.ACTION_DIM), requires_grad=False, device=config.DEVICE)
            obs_cond = torch.randn((1, config.IMG_EMBEDDING_DIM), requires_grad=False, device=config.DEVICE)
            k = torch.tensor(0, requires_grad=False, device=config.DEVICE)
            torch_pred = noise_pred_net(sample=noisy_action,
                        timestep=k,
                        global_cond=obs_cond)
            _ = noise_scheduler.step(
                model_output=torch_pred,
                timestep=k,
                sample=noisy_action
            ).prev_sample
        end_normal_time = time.perf_counter()
        total_time = end_normal_time - start_normal_time
        avg_time_per_iter = (total_time * 1000) / n
        print(f"Completed {n} normal mode evaluations in {total_time:.4f} seconds (avg {avg_time_per_iter:.4f} ms per iter).") 


def unet_inference_eval_accelerated(n, noise_pred_net):
    """
    Performs inference evaluation in accelerated mode for a given number of iterations.

    This function evaluates the inference time of the noise prediction network in PyTorch accelerated mode. It generates random
    noisy actions and observation conditions, then uses the preloaded CUDA graph of the noise prediction network to denoise the actions.
    The function measures and prints the total and average time taken for the specified number of iterations.

    Parameters:
    - n (int): The number of iterations to run the evaluation.
    - noise_pred_net (torch.nn.Module): The noise prediction network to be evaluated.
    """

    with config.inference_mode_context(config.InferenceMode.ACCELERATED):
        u_net_graph, static_noisy_action, static_k, static_obs_cond, static_diffusion_noise, _ = load_noise_pred_net_graph(noise_pred_net)
        start_accelerated_time = time.perf_counter()
        for _ in tqdm(range(n), desc='Inference Eval - Accelerated Mode'):
            noisy_action = torch.randn((1, config.PRED_HORIZON, config.ACTION_DIM), requires_grad=False, device=config.DEVICE)
            obs_cond = torch.randn((1, config.IMG_EMBEDDING_DIM), requires_grad=False, device=config.DEVICE)
            k = torch.tensor(0, requires_grad=False, device=config.DEVICE)
            diffusion_noise = torch.randn((1, config.PRED_HORIZON, config.ACTION_DIM), device=config.DEVICE)
            static_k.copy_(k)
            static_noisy_action.copy_(noisy_action)
            static_obs_cond.copy_(obs_cond)
            static_diffusion_noise.copy_(diffusion_noise)
            u_net_graph.replay()  
        end_accelerated_time = time.perf_counter()
        total_time = end_accelerated_time - start_accelerated_time
        avg_time_per_iter = (total_time * 1000) / n
        print(f"Completed {n} accelerated mode forward passes in {total_time:.4f} seconds (avg {avg_time_per_iter:.4f} ms per iter).") 

def diffusion_policy_eval(n, env_handler, model_manager, mode=config.InferenceMode.NORMAL):
    """
    Evaluates the diffusion policy model in either normal or accelerated inference mode.

    This function orchestrates the evaluation of the diffusion policy model by setting the inference mode, initializing the environment and model manager, 
    and running the evaluation loop for a specified number of episodes. It measures and prints the total and average reward across all episodes.

    Parameters:
    - n (int): The number of episodes to run the evaluation.
    - env_handler (MDPEnvironment): The environment handler for interacting with the simulation environment.
    - model_manager (ModelManager): The manager for loading and running the diffusion policy model.
    - mode (InferenceMode): The mode of inference to use, either NORMAL or ACCELERATED.
    """

    with config.inference_mode_context(mode):
        start_time = time.perf_counter()
        rewards = []
        desc = f'Diffusion Policy Eval - {"Pytorch Eager" if mode == config.InferenceMode.NORMAL else "Accelerated"} Mode'
        for _ in tqdm(range(n), desc=desc):
            config.reset_seed()
            ep_reward = run_evaluation(env_handler, model_manager)
            rewards.append(ep_reward)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_reward = sum(rewards) / n
        print(f"Completed {n} {mode.value} mode evaluations in {total_time:.4f} seconds (avg {total_time / n:.4f} seconds per episode).")
        print(f"Average reward per episode: {avg_reward:.4f}")

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Run evaluations with or without inference acceleration.")

    # Add arguments
    parser.add_argument('--mode', type=str, choices=['inference-eval', 'policy-eval'], default='inference-eval',
                        help="Choose the evaluation mode: 'inference-eval' for UNet Inference Evaluation or 'policy-eval' for Diffusion Policy Evaluation. Default is 'inference-eval'.")
    parser.add_argument('--evals', type=int, default=None,
                        help="Specify the number of evaluations to run. Defaults are 100 for 'inference-eval' and 5 for 'policy-eval'.")

    # Parse the arguments
    args = parser.parse_args()

    # Set default evals based on the mode if not specified
    if args.evals is None:
        args.evals = 1000 if args.mode == 'inference-eval' else 5

    if args.mode == "inference-eval":
        noise_pred_net = ConditionalUnet1D(
            input_dim=config.ACTION_DIM,
            global_cond_dim=config.IMG_EMBEDDING_DIM
        ).to(config.DEVICE)
        unet_inference_eval_eager(args.evals, noise_pred_net)
        unet_inference_eval_accelerated(args.evals, noise_pred_net)
    elif args.mode == "policy-eval":
        model_manager = ModelManager()
        env_handler = MDPEnvironment()
        diffusion_policy_eval(args.evals, env_handler, model_manager, config.InferenceMode.NORMAL)
        diffusion_policy_eval(args.evals, env_handler, model_manager, config.InferenceMode.ACCELERATED)

if __name__ == "__main__":
    main()