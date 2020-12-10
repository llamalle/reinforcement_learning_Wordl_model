from simple_playgrounds import Engine
from simple_playgrounds.playgrounds import SingleRoom
from simple_playgrounds.entities.texture import UniformTexture, CenteredRandomTilesTexture, PolarStripesTexture, RandomTilesTexture
from simple_playgrounds.utils import PositionAreaSampler
from simple_playgrounds.entities.scene_elements import Basic
from simple_playgrounds.entities.agents import BaseAgent, HeadAgent
from simple_playgrounds.entities.agents.sensors import DepthSensor, ProximitySensor, RgbSensor
from simple_playgrounds.controllers import Keyboard, Random
import cv2
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random
import copy

SHOW_ENVIRONMENT = False
SHOW_HEATMAPS = False
SAVE_SIMU = True
SCALE = 1.25

def set_randoom_position():
    positions = [
        [35,270,0],
        [35,150,0],
        [35,30,0],
        [250,30,0],
        [150,30,0],
        [250,250,0],
        [150,30,0]
    ]
    return random.choice(positions)
m=0

my_playground = SingleRoom(size=(int(250*SCALE), int(250*SCALE)), wall_type='light')
# my_agent = HeadAgent(controller=Keyboard())
my_agent = BaseAgent(initial_position=[25,270,-1], name='robot', controller=Random())

IR_sensor = ProximitySensor(my_agent.base_platform, name='IR_1', normalize=False, range=270,fov=270, resolution=1080, point_angle=0.125)

my_agent.add_sensor(IR_sensor)

my_playground.add_agent(my_agent)

engine = Engine(time_limit=80000, playground=my_playground, screen=True)

##################################
# AMENAGEMENT DE L'ENVIRONNEMENT #
##################################

text_uniform_rectangle = UniformTexture(color_min=(255, 50, 0), color_max=(255, 150, 0), size_tiles=int(1*SCALE), radius=int(200))

#rectangular_object1 = Basic((int(150), int(240), 0), physical_shape='rectangle',width_length=(int(3), int(170)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)

#rectangular_object1 = Basic((int(150), int(290), 0), physical_shape='rectangle',width_length=(int(3), int(250)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)

#rectangular_object1 = Basic((int(107), int(185), -1), physical_shape='rectangle',width_length=(int(3), int(130)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)

#rectangular_object1 = Basic((int(47), int(185), -1.1), physical_shape='rectangle',width_length=(int(3), int(130)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)

#rectangular_object1 = Basic((int(294), int(210), 0), physical_shape='rectangle',width_length=(int(160), int(3)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)

#rectangular_object1 = Basic((int(245), int(190), 0), physical_shape='rectangle',width_length=(int(110), int(3)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)

#rectangular_object1 = Basic((int(55), int(70), -0.5), physical_shape='rectangle',width_length=(int(120), int(3)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(20), int(150), 0), physical_shape='rectangle',width_length=(int(280), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(100), int(180), 0), physical_shape='rectangle',width_length=(int(220), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)


#rectangular_object1 = Basic((int(120), int(90), -0.5), physical_shape='rectangle',width_length=(int(80), int(3)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(150), int(15), 0), physical_shape='rectangle',width_length=(int(3), int(260)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(160), int(73), 0), physical_shape='rectangle',width_length=(int(3), int(125)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(295), int(150), 0), physical_shape='rectangle',width_length=(int(250), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = Basic((int(220), int(180), 0), physical_shape='rectangle',width_length=(int(210), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

#rectangular_object1 = Basic((int(270), int(135), 0), physical_shape='rectangle',width_length=(int(3), int(45)), texture=text_uniform_rectangle)
#my_playground.add_scene_element(rectangular_object1)


#################################
# SIMULATION ET RECOLTE DONNEES #
#################################

all_observations = {}
all_actions = {}
all_observations[my_agent.name] = {}
all_actions[my_agent.name] = {}
for part in my_agent.parts:
    all_actions[my_agent.name][part.name] = {}

if SHOW_HEATMAPS:
    heatMat = np.zeros((my_playground.length, my_playground.width, 7))

stop = True 

while engine.game_on and stop:

    if SHOW_HEATMAPS:
        i, j, k = int(my_agent.position[0]), int(my_agent.position[1]), int(my_agent.position[2])
        heatMat[i, j, k] += 1

    print(str(engine.elapsed_time) + '/' + str(engine.time_limit))

    if SHOW_ENVIRONMENT:
        engine.display_full_scene()

    actions = {}
    actions[my_agent.name] = my_agent.controller.generate_actions()
    if engine.elapsed_time == 0 and SAVE_SIMU:
        for part in my_agent.parts:
            available_actions = part.get_available_actions()
            for action in available_actions:
                all_actions[my_agent.name][part.name][action.action.name] = [actions[my_agent.name][
                                                                                part.name][action.action]]
    elif engine.elapsed_time > 0 and SAVE_SIMU:
        for part in my_agent.parts:
            available_actions = part.get_available_actions()
            for action in available_actions:
                all_actions[my_agent.name][part.name][action.action.name].append(actions[my_agent.name][
                                                                                    part.name][action.action])

    if engine.elapsed_time == 0:  # Pour contourner un petit bug temporaire
        engine.step(actions)
    else:
        engine.multiple_steps(actions, n_steps=2)

    if SAVE_SIMU or SHOW_ENVIRONMENT:
        engine.update_observations()
        observation = IR_sensor.sensor_value


    if SAVE_SIMU:
        for sensor in my_agent.sensors:
            observation = sensor.sensor_value
            sensor_name = sensor.name
            if engine.elapsed_time == 1:
                all_observations[my_agent.name][sensor_name] = [observation]
            else:
                all_observations[my_agent.name][sensor_name].append(observation)

    if SHOW_ENVIRONMENT:
        cv2.imshow('sensor', engine.generate_sensor_image(my_agent))
        cv2.waitKey(20)

    if (engine.elapsed_time % 10001 == 0 or engine.elapsed_time == 119999) and SHOW_HEATMAPS:
        heatMatAllDirections = np.zeros((my_playground.length, my_playground.width))
        heatMatCumulative = np.zeros((my_playground.length, my_playground.width))
        for theta in range(np.shape(heatMat)[2]):
            heatMatAllDirections += heatMat[:, :, theta] > 0 * 1
            # heatMatCumulative += heatMat[:, :, theta]
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.rot90(heatMatCumulative), cmap='gray')
        # plt.title("Cumul total")
        # plt.subplot(1, 2, 2)
        plt.imshow(np.rot90(heatMatAllDirections), cmap='gray')
        plt.title("T="+str((engine.elapsed_time-1)/2))
        plt.subplots_adjust(left=0.05, bottom=0.05, top=0.95, right=0.98)
        # plt.savefig("HeatMap_STEP="+str(int((engine.total_elapsed_time-1)/2)),dpi=400)
        plt.show()
    
    if engine.elapsed_time%5000 == 1:
        my_agent.position = set_randoom_position()

    if engine.elapsed_time >= 79999:
        stop = False

    
    print(np.array(list(actions[my_agent.name]['base'].values())))

#engine.terminate()
#cv2.destroyAllWindows()

if SAVE_SIMU:
    fid = open(
        "data/new_recordings.p", "wb")
    pickle.dump({'observations': all_observations, 'actions': all_actions}, fid)
