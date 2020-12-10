import torch
import torch.nn as nn

class brain_controller(nn.Module):

    def __init__(self, w):
        super(brain_controller, self).__init__()
        self.m = nn.Linear(32,3)
        self.m.weight.data = w
    