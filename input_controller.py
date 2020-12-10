import torch
import numpy as np

def compute_input(encoder, gru, observation, previous_observations, previous_actions):
    return torch.from_numpy(np.random.uniform(-1,1,[3,64])).float()