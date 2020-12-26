from simple_playgrounds.controllers import Keyboard, Random, External
from simple_playgrounds.utils.definitions import ActionTypes, KeyTypes
from abc import ABC, abstractmethod
import random
import pygame
from pygame.locals import *
import torch.nn as nn

from simple_playgrounds.utils.definitions import ActionTypes, KeyTypes

class my_controller(External):
    def __init__(self, w, input_size):
        super(my_controller, self).__init__()
        self.model = nn.Linear(input_size,3)
        self.model.weight.data = w

    

    def generate_actions(self, observation):
        A = self.model(observation)
        actions = self.null_actions.copy()
        """for action in self.available_actions:
            if action.action_type == ActionTypes.CONTINUOUS_CENTERED:
                act_value = A[0][0]

            elif action.action_type == ActionTypes.CONTINUOUS_NOT_CENTERED:
                act_value =  A[0][1]

            elif action.action_type == ActionTypes.DISCRETE:
                act_value = A[0][2]

            else:
                raise ValueError

            actions[action.body_part][action.action] = act_value"""
        i=0
        for action in self.available_actions:
                actions[action.body_part][action.action] = A[0][i].data/16
                i+=1

        return actions
