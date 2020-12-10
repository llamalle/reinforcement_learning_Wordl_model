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
sys.path.insert(1,"/home/louis/projU2IS/controller/rnn")

from population import Population, createPopulation
from individual import Individual, ParamIndividual
from my_controller import my_controller
from vaes import VAELow, val_vae, train_vae, losses_training
from gru import GRUNet
from input_controller import compute_input

SHOW_ENVIRONMENT = False
SHOW_HEATMAPS = False
SAVE = True
SCALE = 1.25
R=0.5
RT=0.5
NUMBER_GEN = 25
SIZE = 20
MIN = 9.908510221828081
MAX = 270

def get_key(individual):
    return individual.reward

def plt_image(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    

my_playground = SingleRoom(size=(int(250*SCALE), int(250*SCALE)), wall_type='light')

############LOAD VAE AND GRU#################
encoder = VAELow()
name_vae = "trained_vaes/VAELow_new.pt"
state_dict_VAE = torch.load(name_vae)['state_dict']
encoder.load_state_dict(state_dict_VAE)

gru = GRUNet(seq_in=16, seq_out=3,  n_layers=1, drop_prob=0.0, input_action_dim=3,
                    input_states_dim=32, output_dim=32)
name_gru = "trained_rnns/myGRU_new.pt"
state_dict_GRU = torch.load(name_gru)['state_dict']
gru.load_state_dict(state_dict_GRU)

##################################
# AMENAGEMENT DE L'ENVIRONNEMENT #
##################################
text_uniform_rectangle = UniformTexture(color_min=(255, 50, 0), color_max=(255, 150, 0), size_tiles=int(1*SCALE), radius=int(200))

rectangular_object1 = VisibleDeathTrap((int(20), int(150), 0), physical_shape='rectangle',width_length=(int(280), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = VisibleDeathTrap((int(100), int(180), 0), physical_shape='rectangle',width_length=(int(220), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = VisibleDeathTrap((int(150), int(15), 0), physical_shape='rectangle',width_length=(int(3), int(260)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = VisibleDeathTrap((int(160), int(73), 0), physical_shape='rectangle',width_length=(int(3), int(125)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = VisibleDeathTrap((int(295), int(150), 0), physical_shape='rectangle',width_length=(int(250), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)

rectangular_object1 = VisibleDeathTrap((int(220), int(180), 0), physical_shape='rectangle',width_length=(int(210), int(3)), texture=text_uniform_rectangle)
my_playground.add_scene_element(rectangular_object1)


##########################ADD REWARD ZONE######################################

RZ = HealingZone([60,250,0], physical_shape="rectangle", width_length=(10,70), reward=R, total_reward=RT)
my_playground.add_scene_element(RZ)

RZ2 = HealingZone([60,230,0], physical_shape="rectangle", width_length=(10,70), reward=R, total_reward=RT)
my_playground.add_scene_element(RZ2)

RZ3 = HealingZone([60,210,0], physical_shape="rectangle", width_length=(10,70), reward=R, total_reward=RT)
my_playground.add_scene_element(RZ3)

RZ4 = HealingZone([60,190,0], physical_shape="rectangle", width_length=(10,70), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ4)

RZ5 = HealingZone([60,170,0], physical_shape="rectangle", width_length=(10,70), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ5)

RZ6 = HealingZone([60,150,0], physical_shape="rectangle", width_length=(10,70), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ6)

RZ7 = HealingZone([60,130,0], physical_shape="rectangle", width_length=(10,70), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ7)

RZ8 = HealingZone([60,110,0], physical_shape="rectangle", width_length=(10,70), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ8)

RZ9 = HealingZone([60,90,0], physical_shape="rectangle", width_length=(10,70), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ9)

RZ10 = HealingZone([60,70,0], physical_shape="rectangle", width_length=(10,70), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ10)


RZ12 = HealingZone([65,40,0], physical_shape="square", radius=28, reward=R,total_reward=RT)
my_playground.add_scene_element(RZ12)

RZ14 = HealingZone([125,45,0], physical_shape="rectangle", width_length=(50,10), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ14)

RZ15 = HealingZone([145,45,0], physical_shape="rectangle", width_length=(50,10), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ15)

RZ16 = HealingZone([165,45,0], physical_shape="rectangle", width_length=(50,10),reward=R, total_reward=RT)
my_playground.add_scene_element(RZ16)

RZ17 = HealingZone([185,45,0], physical_shape="rectangle", width_length=(50,10), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ17)

RZ18 = HealingZone([205,45,0], physical_shape="rectangle", width_length=(50,10),reward=R, total_reward=RT)
my_playground.add_scene_element(RZ18)

RZ19 = HealingZone([225,45,0], physical_shape="rectangle", width_length=(50,10),reward=R, total_reward=RT)
my_playground.add_scene_element(RZ19)

RZ20 = HealingZone([245,45,0], physical_shape="rectangle", width_length=(50,10), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ20)

RZ21 = HealingZone([265,45,0], physical_shape="rectangle", width_length=(50,10), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ21)

RZ22 = HealingZone([257,90,0], physical_shape="rectangle", width_length=(10,60), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ22)

RZ23 = HealingZone([257,110,0], physical_shape="rectangle", width_length=(10,60),reward=R, total_reward=RT)
my_playground.add_scene_element(RZ23)

RZ24 = HealingZone([257,130,0], physical_shape="rectangle", width_length=(10,60),reward=R, total_reward=RT)
my_playground.add_scene_element(RZ24)

RZ25 = HealingZone([257,150,0], physical_shape="rectangle", width_length=(10,60), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ25)

RZ26 = HealingZone([257,170,0], physical_shape="rectangle", width_length=(10,60), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ26)

RZ27 = HealingZone([257,190,0], physical_shape="rectangle", width_length=(10,60),reward=R, total_reward=RT)
my_playground.add_scene_element(RZ27)

RZ28 = HealingZone([257,210,0], physical_shape="rectangle", width_length=(10,60), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ28)

RZ29 = HealingZone([257,230,0], physical_shape="rectangle", width_length=(10,60), reward=R,total_reward=RT)
my_playground.add_scene_element(RZ29)

##############################BLOCK ENTRY###############################

End = VisibleEndGoal([260,280,0], physical_shape="rectangle", width_length=(10,60), reward=R,total_reward=RT)
my_playground.add_scene_element(End)

#################################INITIALIZE AGENT#######################################
"""my_agent = BaseAgent(initial_position=[15,270,0], name='robot', controller=Keyboard())
my_playground._add_agent(my_agent)
engine = Engine(time_limit=120000, playground=my_playground, screen=True)"""

#############################CREATE 1ST GENERATION#################################
pop0 = Population(SIZE, 0)
for i in range(SIZE):
    individual = ParamIndividual(torch.from_numpy(np.random.uniform(-1,1,[3,64])))
    pop0.addIndividual(individual)
    print(individual.weight)


pop = pop0

best_agent = np.zeros([3,32])
best_reward = -400
best_time = 1200000
mean_rewards = []
###################################################################################

for i in range(NUMBER_GEN):
    #######CREER GENEARATION ###############
    if i != 0:
        pop = createPopulation(SIZE, i+1, previous_pop, 64)

    #######SET AGENT, SNESOR, ENGINE########
    my_agent = BaseAgent(initial_position=[40,270,0], name='robot', controller=my_controller(pop.pop[0].weight.float(), 64))
    IR_sensor = ProximitySensor(my_agent.base_platform, name='IR_1', normalize=False, range=270,fov=270, resolution=1080, point_angle=0.125)
    my_agent.add_sensor(IR_sensor)
    my_playground.add_agent(my_agent)
    engine = Engine(time_limit=120000, playground=my_playground, screen=True)

    ########SET VARIABLES###########################
    game_on = True
    rewards = []
    totpos=200
    totneg=0
    iteration=0
    time=0
    previous_obs = np.zeros((16,32))
    previous_act = np.zeros((16,3))

    while (game_on):
        time+=1
        #cv2.imshow('sensor', engine.generate_sensor_image(my_agent))
        #cv2.waitKey(20)
        if time>450 or abs(totneg)>totpos+5:
            totneg -= 400

        ###################ENCODE THE OBSERVATION################
        observation = torch.from_numpy(IR_sensor.sensor_value[28:1052])
        observation = (observation-MIN)/(MAX-MIN)
        moy, logvar = encoder.encode(observation)
        encode_observation = encoder.reparametrize(moy, logvar)
        
        with torch.no_grad():
            previous_obs = np.append(previous_obs, encode_observation, axis=0)[1:]
            print(previous_obs)
        #######################################################


        actions = {}
        actions[my_agent.name] = my_agent.controller.generate_actions(compute_input(encoder, gru, encode_observation, encode_observation, encode_observation))
        previous_act = np.append(previous_act, [np.array(list(actions[my_agent.name]["base"].values()))], axis=0)[1:]

        reset, terminate = engine.step(actions)
        engine.update_observations()

        engine.display_full_scene()
        #pygame.time.wait(30)


        if engine.elapsed_time%1000 == 1:
            print("t")

        for agent in engine.agents:
            if agent.reward != 0:
                #print(agent.name, ' got reward ', agent.reward)
                totpos+=agent.reward


            """if steps is not None:
                steps -= 1
                if steps == 0:
                    continue_for_n_steps = False"""

        if (time/25 > totpos-199) and not terminate:
            terminate = True
            totpos -= 200

        if terminate:
            pop.pop[iteration].reward = totpos
            pop.pop[iteration].time = time
            time=0
            totpos=200
            totneg=0
            iteration += 1
            engine.reset()
            my_playground.remove_agent(my_agent)
            
            if iteration < SIZE:
                my_agent = BaseAgent(initial_position=[40,270,0], name='robot', controller=my_controller(pop.pop[iteration].weight.float(), 64))
                IR_sensor = ProximitySensor(my_agent.base_platform, name='IR_1', normalize=False, range=270,fov=270, resolution=1080, point_angle=0.125)
                my_agent.add_sensor(IR_sensor)
                my_playground.add_agent(my_agent)

            else :
                game_on = False
                previous_pop = pop
                previous_pop.pop.sort(key=lambda i: (i.reward,-i.time), reverse=True)
                summ=0
                for k in previous_pop.pop:
                    summ += k.reward
                    print(k.reward)
                print("mean reward", summ/SIZE)
                mean_rewards.append(summ/SIZE)
                print("nouvelle gen")
                
                if previous_pop.pop[0].reward >= best_reward:
                    """if previous_pop.pop[0].reward == best_reward:
                        if previous_pop.pop[0].time < best_time:
                            best_reward = previous_pop.pop[0].reward
                            best_agent = previous_pop.pop[0].weight
                            best_time = previous_pop.pop[0].time"""
            
                    best_reward = previous_pop.pop[0].reward
                    best_agent = previous_pop.pop[0].weight
                    best_time = previous_pop.pop[0].time
                ###############sort pop by reward##############
                print("best reward", best_reward, "best time", best_time)
                ######################################

mean_rewards = np.array(mean_rewards)

if SAVE :
    print(best_agent)
    with open("best_agent2.npy", "wb") as f:
        np.save(f, best_agent)

    with open("means_trace.npy", "wb") as f2:
        np.save(f2, mean_rewards)
