import sys
sys.path.insert(1,"individual.py")
from individual import RandomIndividual, ParamIndividual
import random
import torch
import numpy as np
import copy
from copy import copy, deepcopy

class Population():
    def __init__(self, size, generation_number):
        self.pop = []
        self.size = size
        self.generation_number = generation_number

    def addIndividual(self, individual):
        self.pop.append(individual)

class RandomPopulation(Population):
    def __init__(self,size,generation_number):
        super(RandomPopulation, self).__init__(size, generation_number)


def cross(selected_individuals, input_size):
    A, B = selected_individuals
    A.weight = torch.flatten(A.weight)
    B.weight = torch.flatten(B.weight)
    i = random.choice(range(len(A.weight))) 
    j = random.choice(range(len(A.weight)))
    i, j = min(i,j), max(i,j)  
    
    tmp = torch.clone(A.weight[i:j])
    A.weight[i:j] = torch.clone(B.weight[i:j])
    B.weight[i:j] = torch.clone(tmp)

    A.weight = torch.reshape(A.weight,(3,input_size))
    B.weight = torch.reshape(B.weight,(3,input_size))
 
    return A,B

def mutation(individual, input_size):
    min = torch.min(individual.weight)
    max = torch.max(individual.weight)
    new_weight = random.uniform(min, max)
    i = random.randint(0,2)
    j = random.randint(0,input_size-1)
    individual.weight[i][j] = new_weight
    print("mutation on ")
    print(individual, i, j)
    return individual

def createRandomPopulation(size, generation_number):
    population = Population(size, generation_number)
    population.pop = [ParamIndividual(torch.from_numpy(np.random.rand(3,32))) for i in range(size)]
    return population


def createPopulation(size, generation_number, previous_generation, input_size):
    probas = [1/(i+5) for i in range(size)]
    population = Population(size, generation_number)
    
    for i in previous_generation.pop[:int(size/5)]:
        population.pop.append(i)

    for _ in range(int(size/5)):
        individual = ParamIndividual(torch.from_numpy(np.random.uniform(-1,1,[3,input_size])))
        population.pop.append(individual)

    while(len(population.pop) < size):
        A,B = random.choices(previous_generation.pop, weights=probas, k=2)
        if random.randint(0,100) > 30:
            A,B = cross([A,B], input_size)
        
        for i in range(2):
            if random.randint(0, 1000) > 500:
                B = mutation(B, input_size)
            if random.randint(0, 1000) > 500:
                A = mutation(A, input_size)
        

        population.pop.append(deepcopy(A))
        population.pop.append(deepcopy(B))

    return population

    