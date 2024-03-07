from enum import Enum
import random 
from contextlib import contextmanager

import torch 
import numpy as np


class InferenceMode(Enum):
    NORMAL = 'normal'
    ACCELERATED = 'accelerated'

@contextmanager
def inference_mode_context(new_mode):
    '''
    Sets global inference mode using a context manager. 
    '''
    global INFERENCE_MODE
    old_mode = INFERENCE_MODE
    INFERENCE_MODE = new_mode
    try:
        yield
    finally:
        INFERENCE_MODE = old_mode

def reset_seed():
    global seed #we could make config a singleton class to avoid modifying global state but this library is pretty tiny so I chose to avoid excessive OOP 
    new_seed = random.randint(2000, 1000000)
    seed = new_seed
    torch.manual_seed(new_seed)

INFERENCE_MODE = InferenceMode.NORMAL
DEVICE = torch.device('cuda')
VISION_FEATURE_DIM = 512
LOWDIM_OBS_DIM = 2
OBS_DIM = VISION_FEATURE_DIM + LOWDIM_OBS_DIM
ACTION_DIM = 2
PRED_HORIZON = 16
OBS_HORIZON = 2
ACTION_HORIZON = 8
NUM_DIFFUSION_ITERS = 100
SEED = 731642#random.randint(2000, 1000000) #731642, 957996, 602326, 332768, 152876
MAX_STEPS = 200
IMG_EMBEDDING_DIM = 1028
STATS = { #hardocded normalization stats to eliminate dataset dependency
    'agent_pos': {
        'min': np.array([13.456424, 32.938293], dtype=np.float32), 
        'max': np.array([496.14618, 510.9579], dtype=np.float32)
    }, 
    'action': {
        'min': np.array([12., 25.], dtype=np.float32), 
        'max': np.array([511., 511.], dtype=np.float32)
    }
}


TENSOR_SHAPES = \
[
(256,256, 16), 
(2048,512, 4), 
(1024,1024, 4), 
(1024,1024, 4), 
(256,256, 16), 
(256,256, 8), 
(1024,256, 8), 
(256,256, 8), 
(1024,1024, 4), 
(256,512, 8), 
(512,512, 4), 
(256,256, 8), 
(512,512, 8), 
(256,256, 16), 
(1024,1024, 4), 
(512,512, 8), 
(512,512, 8), 
(1024,1024, 4),  
(1024,1024, 4), 
(256,256, 16), 
(512,512, 4), 
(512,1024, 4), 
(1024,1024, 4), 
(512,512, 4)
]