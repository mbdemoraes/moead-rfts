from common.problem import Problem
from algorithms.moead_rfts import Moead_Rfts
import random

"""
Contains a working example of MOEA/D-RFTS on the Binary Multi-Objective Unconstrained
Combinatorial Optimization Problem (BIN_MUCOP). Fore more information on the algorithm and the test
function, please refer to the paper

M. B. De Moraes and G. P. Coelho, "A Random Forest-Assisted
Decomposition-Based Evolutionary Algorithm for Multi-Objective
Combinatorial Optimization Problems,"
2022 IEEE Congress on Evolutionary Computation (CEC), 2022,
pp. 1-8, doi: 10.1109/CEC55065.2022.9870412.

"""

#optimization parameters
num_of_variables = 100
num_of_individuals = 100
generations = 50
directions = ["max", "max"]

#Get the profit vector from the instance file
def get_profit_vector(number, num_of_variables):
    profits = []
    with open("instances/bin_mucop_" + str(num_of_variables) + "_" + str(number) + "_.txt", "rt") as f:
        for line in f:
            currentline = line.split(",")
            profits.append(int(currentline[0]))
    return profits

# Calculates the individual profit based on profit assigned
# for that particular position in that particular knapsack (given by the profits vector)
def get_individual_profit(individual, profits):
    weight, profit = 0, 0
    for (item, data) in zip(individual, profits):
        if item != 0:
            profit += data

    return profit

# F1 function: calculates the profits of an individual based on the first instance
def f1(individual):

    profits = get_profit_vector(number=0, num_of_variables=num_of_variables)
    return get_individual_profit(individual,profits)

# F2 function: calculates the profits of an individual based on the second instance
def f2(individual):
    profits = get_profit_vector(number=1, num_of_variables=num_of_variables)
    return get_individual_profit(individual,profits)



# Class to control the problem parameters
problem = Problem(num_of_variables=num_of_variables,
                  num_of_individuals=num_of_individuals,
                  objectives=[f1, f2],
                  variables_range=[0, 1],
                  mutation=(1/num_of_variables),
                  expand=False,
                  num_of_generations=generations,
                  directions=directions)

#MOEA/D-RFTS hyper-parameters
num_of_neighborhoods = 10
criterion = "squared_error"
max_depth = None
max_features = 1.0
min_samples_leaf = 1
n_estimators = 100
min_samples_split = 2
max_samples = 1.0

random.seed()

#Calls the algorithm and sets the parameters
iteration = Moead_Rfts(problem=problem,
                       num_of_neighborhoods=num_of_neighborhoods,
                       criterion=criterion,
                       max_depth=max_depth,
                       max_features=max_features,
                       min_samples_leaf=min_samples_leaf,
                       n_estimators=n_estimators,
                       min_samples_split=min_samples_split,
                       max_samples=max_samples)
#Runs the optimization
iteration.run()








