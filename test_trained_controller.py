from simple_playgrounds import Engine
from simple_playgrounds.playgrounds import SingleRoom
from simple_playgrounds.entities.texture import UniformTexture, CenteredRandomTilesTexture, PolarStripesTexture, RandomTilesTexture
from simple_playgrounds.utils import PositionAreaSampler
from simple_playgrounds.entities.scene_elements import Basic, VisibleDeathTrap, RewardZone, HealingZone, VisibleEndGoal
from simple_playgrounds.entities.agents import BaseAgent, HeadAgent
from simple_playgrounds.entities.agents.sensors import DepthSensor, ProximitySensor, RgbSensor
from simple_playgrounds.controllers import Keyboard, Random
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random
import pygame
import torch
import sys
sys.path.insert(1,"/home/louis/projU2IS/controller/brain_controller.py")
sys.path.insert(1,"/home/louis/projU2IS/controller/individual.py")
sys.path.insert(1,"/home/louis/projU2IS/controller/population.py")
sys.path.insert(1,"/home/louis/projU2IS/controller/vae")

from population import Population, createPopulation
from individual import Individual, ParamIndividual
from my_controller import my_controller
from vaes import VAELow, val_vae, train_vae, losses_training


MIN = 9.908510221828081
MAX = 270

def plt_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()

with open("means_trace", "rb") as f:
    trace = np.load(f)

plt.plot(trace)
plt.show()

with open("best_agent2.npy", "rb") as f:
    weight = np.load(f)

best_agent = ParamIndividual(torch.from_numpy(weight))
print(best_agent.weight)

my_playground = SingleRoom(size=(int(250*1.25), int(250*1.25)), wall_type='light')
encoder = VAELow()
name = "trained_vaes/VAELow_new.pt"
state_dict_VAE = torch.load(name)['state_dict']
encoder.load_state_dict(state_dict_VAE)

######################ENV###########################
text_uniform_rectangle = UniformTexture(color_min=(255, 50, 0), color_max=(255, 150, 0), size_tiles=int(1*1.25), radius=int(200))

rectangular_object1 = Basic((int(20), int(150), 0), physical_shape='rectangle',width_length=(int(280), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(100), int(180), 0), physical_shape='rectangle',width_length=(int(220), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(150), int(15), 0), physical_shape='rectangle',width_length=(int(3), int(260)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(160), int(73), 0), physical_shape='rectangle',width_length=(int(3), int(125)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(295), int(150), 0), physical_shape='rectangle',width_length=(int(250), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(220), int(180), 0), physical_shape='rectangle',width_length=(int(210), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)
####################################################

my_agent = BaseAgent(initial_position=[40,270,0], name='robot', controller=my_controller(best_agent.weight.float()))
IR_sensor = ProximitySensor(my_agent.base_platform, name='IR_1', normalize=False, range=270,fov=270, resolution=1080, point_angle=0.125)
my_agent.add_sensor(IR_sensor)
my_playground.add_agent(my_agent)
engine = Engine(time_limit=120000, playground=my_playground, screen=True)

while (engine.game_on):

    ###################ENCODE THE OBSERVATION################
    observation = torch.from_numpy(IR_sensor.sensor_value[28:1052])
    observation = (observation-MIN)/(MAX-MIN)
    moy, logvar = encoder.encode(observation)
    encode_observation = encoder.reparametrize(moy, logvar)
    #######################################################

    actions = {}
    actions[my_agent.name] = my_agent.controller.generate_actions(encode_observation.float())

    reset, terminate = engine.step(actions)
    engine.update_observations()

    engine.display_full_scene()
    #pygame.time.wait(30)


    if terminate:
        engine.terminate()