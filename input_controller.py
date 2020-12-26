import torch
import numpy as np

def compute_input(encoder, gru, observation, previous_observations, previous_actions, hidden_state):
    last_obs = torch.from_numpy(previous_observations[-1])
    previous_obs = torch.from_numpy(previous_observations[:-1]).reshape((1,16,32))
    previous_act_in = torch.from_numpy(previous_actions[:-1]).reshape((1,16,3))
    previous_act_out = torch.from_numpy(previous_actions[-1]).reshape((1,1,3))

    prediction, h = gru(hidden_state, previous_obs, previous_act_in, previous_act_out)
    prediction = prediction.reshape((32))
    input = torch.cat((last_obs, prediction))
    input = input.reshape((1,64))
    return input.float(), h