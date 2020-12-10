import torch
import random

class Individual():
    def __init__(self):
        self.weight = []
        self.reward = 0
        self.time = 0

class RandomIndividual(Individual):
    def __init__(self):
        super(RandomIndividual, self).__init__()
        self.weight = torch.randn([3,32])

class ParamIndividual(Individual):
    def __init__(self, w):
        super(ParamIndividual, self).__init__()
        self.weight = w