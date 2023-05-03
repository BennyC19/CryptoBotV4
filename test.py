from abc import ABCMeta, abstractstaticmethod
from collections import deque
import threading
import time
from time import sleep
from datetime import datetime, timedelta
import requests
import json
import hmac
import hashlib
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pygad.torchga
import pygad

counter = 0

def fitness_func(solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function, counter

    predictions = pygad.torchga.predict(model=model,
                                        solution=solution,
                                        data=data_inputs)

    abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001

    solution_fitness = 1.0 / abs_error
    counter += 1
    print(counter)
    return solution_fitness

def callback_generation(ga_instance):
    #print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    #print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    """"""
# Create the PyTorch model.
input_layer = torch.nn.Linear(3, 5)
relu_layer = torch.nn.ReLU()
output_layer = torch.nn.Linear(5, 1)

model = torch.nn.Sequential(input_layer,
                            relu_layer,
                            output_layer)
# print(model)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = pygad.torchga.TorchGA(model=model,
                           num_solutions=10)

loss_function = torch.nn.L1Loss()

# Data inputs
data_inputs = torch.tensor([[0.02, 0.1, 0.15],
                            [0.7, 0.6, 0.8],
                            [1.5, 1.2, 1.7],
                            [3.2, 2.9, 3.1]])

# Data outputs
data_outputs = torch.tensor([[0.1],
                             [0.6],
                             [1.3],
                             [2.5]])

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/README_pygad_ReadTheDocs.html#pygad-ga-class
num_generations = 100 # Number of generations.
num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
#print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
#print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# Make predictions based on the best solution.
predictions = pygad.torchga.predict(model=model,
                                    solution=solution,
                                    data=data_inputs)
#print("Predictions : \n", predictions.detach().numpy())

abs_error = loss_function(predictions, data_outputs)
#print("Absolute Error : ", abs_error.detach().numpy())

"""
        newTrainingData = []
        training_data_array = numpy.array(self.training_data)
        startSlice = 0
        endSlice = 1440
        while endSlice < len(self.training_data):
            training_data_array_slice = training_data_array[startSlice:endSlice]
            newTrainingData.append(list(training_data_array_slice))
            startSlice += 1
            endSlice += 1
        
        self.new_training_data = numpy.array(newTrainingData)

        while self.endIndex < len(self.new_training_data):

            torch_ga = pygad.torchga.TorchGA(model=self.NeuralNet, num_solutions=100)

            ga_instance = pygad.GA(num_generations=10,
                                num_parents_mating=2,
                                initial_population=torch_ga.population_weights,
                                fitness_func=self.fitness_func,
                                parent_selection_type="sss",
                                crossover_type="single_point",
                                mutation_type="random",
                                mutation_percent_genes=10,
                                keep_parents=-1)

            ga_instance.run()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            model_weights_matrix = pygad.torchga.model_weights_as_matrix(model=self.NeuralNet, weights_vector=solution)
            self.NeuralNet.set_weights(weights=model_weights_matrix)
"""