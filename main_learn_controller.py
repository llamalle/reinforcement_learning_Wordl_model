import torch
import sys
sys.path.insert(1,"/home/louis/projU2IS/controller/brain_controller.py")
sys.path.insert(1,"/home/louis/projU2IS/controller/individual.py")
sys.path.insert(1,"/home/louis/projU2IS/controller/population.py")
from brain_controller import brain_controller
from population import createRandomPopulation, createPopulation, cross, Population
import random
from termcolor import colored
from individual import RandomIndividual, ParamIndividual
import time
import numpy as np
import copy
from copy import copy, deepcopy


w = torch.from_numpy(np.random.rand(3,32))

B = brain_controller(w.float())
C = brain_controller(w.float())
D = brain_controller(w.float())
E = brain_controller(w.float())


with torch.no_grad():
    print(B.m(e.float()))
    print(C.m(e.float()))
    print(D.m(e.float()))
    print(E.m(e.float()))


