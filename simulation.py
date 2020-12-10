import sys
sys.path.insert(1, '/home/louis/projU2IS/controller/my_controller.py')

from my_controller import my_controller
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

SHOW_ENVIRONMENT = True
SHOW_HEATMAPS = False
SAVE_SIMU = True
SCALE = 1.25

my_playground = SingleRoom(size=(int(250*SCALE), int(250*SCALE)), wall_type='light')
# my_agent = HeadAgent(controller=Keyboard())
m = my_controller()
my_agent = BaseAgent(name='robot', controller=m)

IR_sensor = ProximitySensor(my_agent.base_platform, name='IR_1', normalize=True, range=int(250*SCALE),fov=270, resolution=1080, point_angle=0.125)

my_agent.add_sensor(IR_sensor)

my_playground.add_agent(my_agent)

engine = Engine(time_limit=120000, playground=my_playground, screen=True)

##################################
# AMENAGEMENT DE L'ENVIRONNEMENT #
##################################

text_tiles_circle = RandomTilesTexture(color_min=(0, 0, 150), color_max=(0, 0, 255), size_tiles=int(5*SCALE),
                                       radius=int(40*SCALE))
circular_object = Basic((int(40*SCALE), int(40*SCALE), 0), physical_shape='circle', radius=int(20*SCALE),
                        texture=text_tiles_circle)
my_playground.add_scene_element(circular_object)

circular_object = Basic((int(75*SCALE), int(30*SCALE), 0), physical_shape='circle', radius=int(15*SCALE),
                        texture=text_tiles_circle)
my_playground.add_scene_element(circular_object)

circular_object = Basic((int(25*SCALE), int(60*SCALE), 0), physical_shape='circle', radius=int(10*SCALE),
                        texture=text_tiles_circle)
my_playground.add_scene_element(circular_object)

circular_object = Basic((int(25*SCALE), int(80*SCALE), 0), physical_shape='circle', radius=int(10*SCALE),
                        texture=text_tiles_circle)
my_playground.add_scene_element(circular_object)

circular_object = Basic((int(25*SCALE), int(100*SCALE), 0), physical_shape='circle', radius=int(10*SCALE),
                        texture=text_tiles_circle)
my_playground.add_scene_element(circular_object)

text_uniform_rectangle = UniformTexture(color_min=(255, 50, 0), color_max=(255, 150, 0), size_tiles=int(1*SCALE),
                                        radius=int(40*SCALE))
rectangular_object1 = Basic((int(60*SCALE), int(115*SCALE), -0.1), physical_shape='rectangle',
                            width_length=(int(15*SCALE), int(80*SCALE)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object2 = Basic((int(190*SCALE), int(100*SCALE), 1.8), physical_shape='rectangle',
                            width_length=(int(10*SCALE), int(60*SCALE)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object2)

text_random_tiles_centered = CenteredRandomTilesTexture(color_min=(100, 0, 100), color_max=(200, 0, 200),
                                                        radius=30*SCALE, size_tiles=int(10*SCALE))
square_object = Basic([int(200*SCALE), int(50*SCALE), 0.25], physical_shape='square', radius=int(30*SCALE),
                      texture=text_random_tiles_centered)
my_playground.add_scene_element(square_object)

hexagonal_object = Basic((int(50*SCALE), int(200*SCALE), 0.3), default_config_key='hexagon', radius=int(40*SCALE))
my_playground.add_scene_element(hexagonal_object)

text_uniform_triangle = UniformTexture(color_min=(255, 255, 0), color_max=(0, 255, 0), size_tiles=int(1*SCALE),
                                       radius=int(30*SCALE))
center_area, size_area = my_playground.get_quarter_area((0, 0), 'up-right')
area_all = PositionAreaSampler(center=center_area, area_shape='circle', radius=int(size_area[0]/4*SCALE))
n_triangles = 3
for i in range(n_triangles):
    circular_object = Basic(area_all, physical_shape='triangle', radius=int(8*SCALE), texture=text_uniform_triangle,
                            allow_overlapping=False)
    my_playground.add_scene_element(circular_object)


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
        # cv2.waitKey(20)

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

    if engine.elapsed_time >= 120000:
        stop = False
    

engine.terminate()
cv2.destroyAllWindows()

if SAVE_SIMU:
    fid = open(
        "data/recordings.p", "wb")
    pickle.dump({'observations': all_observations, 'actions': all_actions}, fid)
