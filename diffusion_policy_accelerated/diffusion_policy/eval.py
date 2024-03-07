import collections 
import os

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch 
import torch.nn as nn
import gdown 
import numpy as np
from tqdm import tqdm 

import diffusion_policy_accelerated.config as config 
from diffusion_policy_accelerated.diffusion_policy.push_t_env import PushTImageEnv
from diffusion_policy_accelerated.diffusion_policy.model import get_resnet, replace_bn_with_gn, ConditionalUnet1D, load_noise_pred_net_graph, normalize_data, unnormalize_data

class MDPEnvironment:
    def __init__(self):
        '''
        A class representing the MDP environment for evaluation. 
        Handles keep track of observations, rewards, and changing env states with actions. 
        '''
        self.env = PushTImageEnv()
        self.reset() 

    def reset(self):
        self.env.seed(config.SEED)
        obs, info = self.env.reset()
        self.obs_deque = collections.deque([obs] * config.OBS_HORIZON, maxlen=config.OBS_HORIZON)
        self.rewards = []
        return obs, info

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        self.obs_deque.append(obs)
        self.rewards.append(reward)
        return obs, reward, done, info, max(self.rewards)

class ModelManager:
    '''
    A class that manages the loading and inference of the diffusion policy model. Handles model initialization, weight loading, 
    CUDA graph generation (in accelerated mode) and predicting actions using diffusion. 
    '''
    def __init__(self):
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.NUM_DIFFUSION_ITERS,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        vision_encoder = get_resnet('resnet18')
        self.vision_encoder = replace_bn_with_gn(vision_encoder)

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config.ACTION_DIM,
            global_cond_dim=config.OBS_DIM*config.OBS_HORIZON
        )

        ckpt_path = os.path.join(os.path.dirname(__file__), "pusht_vision_100ep.ckpt")
        if not os.path.isfile(ckpt_path):
            id = "1XKpfNSlwYMGaF5CncoFaLKCDTWoLAHf1&confirm=t"
            gdown.download(id=id, output=ckpt_path, quiet=False)
        state_dict = torch.load(ckpt_path, map_location='cuda')

        self.nets = nn.ModuleDict({
            'vision_encoder': self.vision_encoder,
            'noise_pred_net': self.noise_pred_net
        })
        self.nets.load_state_dict(state_dict)
        self.nets.to(device=config.DEVICE)
        
        with config.inference_mode_context(config.InferenceMode.ACCELERATED):
            self.noise_pred_net_graph, self.static_noisy_action, self.static_k, self.static_obs_cond, \
            self.static_diffusion_noise, self.static_model_output = load_noise_pred_net_graph(self.noise_pred_net)

    def predict_action(self, obs_deque):
        """
        Predicts the next action to take based on the current observation deque.

        This method processes the observation deque to extract image and agent position features, normalizes these features, and then
        uses the vision encoder to generate image embeddings. A random noisy action is then generated and denoised through either
        a normal or accelerated inference process depending on the configured inference mode. The final action is unnormalized before
        being returned.

        Parameters:
        - obs_deque (collections.deque): A deque containing the most recent observations.

        Returns:
        - numpy.ndarray: The predicted action to take.
        """
        images = np.stack([x['image'] for x in obs_deque])
        agent_poses = np.stack([x['agent_pos'] for x in obs_deque])
        nagent_poses = normalize_data(agent_poses, stats=config.STATS['agent_pos'])

        nimages = images
        nimages = torch.from_numpy(nimages).to(config.DEVICE)
        nagent_poses = torch.from_numpy(nagent_poses).to(config.DEVICE)

        with torch.no_grad():
            image_features = self.vision_encoder(nimages)
            obs_features = torch.cat([image_features, nagent_poses], dim=-1)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1).to(config.DEVICE).float()
            noisy_action = torch.randn((1, config.PRED_HORIZON, config.ACTION_DIM), device=config.DEVICE)

            if config.INFERENCE_MODE == config.InferenceMode.NORMAL:
                for k in self.noise_scheduler.timesteps:
                    noise_pred = self.noise_pred_net(
                        sample=noisy_action,
                        timestep=k,
                        global_cond=obs_cond
                    )
                    noisy_action = self.noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=noisy_action
                    ).prev_sample
            else:
                self.static_model_output.copy_(noisy_action)
                for k in self.noise_scheduler.timesteps:
                    diffusion_noise = torch.randn((1, config.PRED_HORIZON, config.ACTION_DIM), device=config.DEVICE)
                    self.static_k.copy_(k)
                    self.static_noisy_action.copy_(self.static_model_output)
                    self.static_obs_cond.copy_(obs_cond)
                    self.static_diffusion_noise.copy_(diffusion_noise)
                    self.noise_pred_net_graph.replay()  

                noisy_action = self.static_model_output    

            naction = noisy_action.detach().to('cpu').numpy()
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=config.STATS['action'])
            return action_pred

def run_evaluation(env_handler, model_manager):
    '''
    Evaluates the model by resetting the environment, executing predicted actions, and collecting rewards until 
    the episode ends or reaches the maximum steps, returning the highest reward.

    Parameters:
    - env_handler (MDPEnvironment): The environment handler to interact with the simulation environment.
    - model_manager (ModelManager): The model manager that predicts actions based on observations.

    Returns:
    - max_rewards (float): The maximum reward achieved during the episode.
    '''
    _, _ = env_handler.reset()
    done = False
    step_idx = 0

    with tqdm(total=config.MAX_STEPS, desc="Eval PushTImageEnv") as pbar:
        while not done and step_idx < config.MAX_STEPS:
            action_pred = model_manager.predict_action(env_handler.obs_deque)
            start = config.OBS_HORIZON - 1
            end = start + config.ACTION_HORIZON
            action = action_pred[start:end,:]

            for i in range(len(action)):
                _, reward, done, _, max_rewards = env_handler.step(action[i])
                
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > config.MAX_STEPS:
                    done = True
                if done:
                    break

    return max_rewards



