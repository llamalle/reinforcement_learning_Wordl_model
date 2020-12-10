from simple_playgrounds import Engine
from simple_playgrounds.playgrounds import SingleRoom
from simple_playgrounds.entities.texture import UniformTexture, CenteredRandomTilesTexture, PolarStripesTexture, RandomTilesTexture
from simple_playgrounds.utils import PositionAreaSampler
from simple_playgrounds.entities.scene_elements import Basic
from simple_playgrounds.entities.agents import BaseAgent, HeadAgent
from simple_playgrounds.entities.agents.sensors import DepthSensor
from simple_playgrounds.controllers import Keyboard, Random
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from simple_playgrounds.entities.scene_elements import VisibleEndGoal, Candy, VisibleDeathTrap
import random

def set_randoom_position():
    positions = [
        [15,270,0],
        [150,270,0],
        [260,270,0],
        [150,20,0],
        [70,150,0],
        [260,150,0],
        [30,20,0],
    ]

    return random.choice(positions)

def plt_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()


SCALE = 1.25

my_playground = SingleRoom(size=(int(250*SCALE), int(250*SCALE)), wall_type='light')
# my_agent = HeadAgent(controller=Keyboard())

engine = Engine(time_limit=120000, playground=my_playground, screen=True)


ini = [15,270,0]
my_agent = BaseAgent(initial_position=ini, name='robot', controller=Random())
IR_sensor = DepthSensor(my_agent.base_platform, name='IR_1', normalize=True, range=int(250*SCALE),
                                fov=270, number=1080, point_angle=0.125)
my_agent.add_sensor(IR_sensor)
my_playground.add_agent(my_agent)

##################################
# AMENAGEMENT DE L'ENVIRONNEMENT #
##################################
text_uniform_rectangle = UniformTexture(color_min=(255, 50, 0), color_max=(255, 150, 0), size_tiles=int(1*SCALE), radius=int(200))

rectangular_object1 = Basic((int(150), int(240), 0), physical_shape='rectangle',width_length=(int(3), int(170)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(150), int(290), 0), physical_shape='rectangle',width_length=(int(3), int(250)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(107), int(185), -1), physical_shape='rectangle',width_length=(int(3), int(130)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(47), int(185), -1.1), physical_shape='rectangle',width_length=(int(3), int(130)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(294), int(210), 0), physical_shape='rectangle',width_length=(int(160), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(245), int(190), 0), physical_shape='rectangle',width_length=(int(110), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(55), int(70), -0.5), physical_shape='rectangle',width_length=(int(120), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(120), int(90), -0.5), physical_shape='rectangle',width_length=(int(80), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(150), int(15), 0), physical_shape='rectangle',width_length=(int(3), int(260)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(190), int(55), 0), physical_shape='rectangle',width_length=(int(3), int(180)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(290), int(35), 0), physical_shape='rectangle',width_length=(int(45), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(270), int(135), 0), physical_shape='rectangle',width_length=(int(3), int(45)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

while engine.game_on:

    #candy3 = Candy([200,250,0])
    #my_playground.add_scene_element(candy3)
    engine.display_full_scene()

    actions = {}
    for agent in engine.agents:
        actions[agent.name] = agent.controller.generate_actions()

    reset, terminate = engine.step(actions)
    engine.update_observations()

    #cv2.imshow('sensor', engine.generate_sensor_image(my_agent))
    #cv2.waitKey(20)
    
    if engine.elapsed_time%5000 == 1:
        my_agent.position = set_randoom_position()

    if reset:
        engine.reset()
    if terminate:
        engine.terminate()


