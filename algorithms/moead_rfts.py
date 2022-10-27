from common.non_dominated_sort import Non_Dominated_Sort
from common.moead_utils import MoeadUtils
import numpy as np
from common.population import Population
from common.offspring_generation import OffspringGeneration
from common.random_forests import RANDOM_FORESTS
import matplotlib.pyplot as plt

class Moead_Rfts:
    """
    Implementation of MOEA/D-RFTS algorithm based on the paper
    M. B. De Moraes and G. P. Coelho, "A Random Forest-Assisted
    Decomposition-Based Evolutionary Algorithm for Multi-Objective
    Combinatorial Optimization Problems,"
    2022 IEEE Congress on Evolutionary Computation (CEC), 2022,
    pp. 1-8, doi: 10.1109/CEC55065.2022.9870412.
    """

    def __init__(self,
                 problem,
                 neighborhood_size,
                 criterion,
                 max_depth,
                 max_features,
                 min_samples_leaf,
                 n_estimators,
                 min_samples_split,
                 max_samples):
        """
        Class constructor
        :param problem: object of the Problem class
        :param neighborhood_size: neighborhood size
        :param n_estimators: number of estimators (trees)
        :param criterion: criterion that defines how to measure the quality of a split
        :param max_depth: maximum depth of a tree
        :param max_features: number of decision variables to consider when building each tree
        :param min_samples_leaf: minimum number of samples to be at leaf node
        :param min_samples_split: minimum number of samples to split a node
        :param max_samples: number of decision vectors (samples) to consider when building each tree
        """

        self.problem = problem
        self.rf = RANDOM_FORESTS(self.problem,
                                 criterion=criterion,
                                 max_depth=max_depth,
                                 max_features=max_features,
                                 min_samples_leaf=min_samples_leaf,
                                 n_estimators=n_estimators,
                                 min_samples_split=min_samples_split,
                                 max_samples=max_samples)
        self.offspring = OffspringGeneration(self.problem)
        self.utils = MoeadUtils(problem)
        self.m = len(self.problem.objectives)
        self.population = None
        self.ndsort = Non_Dominated_Sort()
        self.external_population = Population()
        self.z = None
        self.visited_external = set()
        self.neighborhood_size = neighborhood_size


    def run(self):
        """
        Run the MOEA/D-RFTS algorithm
        :return: None
        """
        #Set the weight vectors
        weights_vectors, self.problem.num_of_individuals= self.utils.simplex_lattice_design(self.problem.num_of_individuals, self.m)

        #Number of individuals in a neighborhood
        T = np.ceil(self.problem.num_of_individuals / self.neighborhood_size)
        T = int(T)

        #Neighborhood definition
        B = self.utils.set_neighborhoods(weights_vectors, self.problem.num_of_individuals, self.m, T)

        #Initial population
        self.population = self.problem.create_initial_population()

        #Get the decision vectors and objective functions of the initial population
        #and use them to train the Random Forest
        train_features = []
        train_labels = []
        for individual in self.population:
            train_features.append(individual.features)
            train_labels.append(individual.objectives)

        self.rf.fit_population(np.array(train_features), np.array(train_labels))
        self.z = self.utils.find_initial_reference_point(self.population)


        plt.ion()
        fig = plt.figure()
        for i in range(self.problem.num_of_generations):
            print("Generation = " + str(i))
            #Creates or updates the tabu list
            self.problem.create_update_tabu_list(self.population)

            #Creates offspring and retrain the RF
            self.offspring.create_children(self.population, B, T, self.z, weights_vectors, rf_object=self.rf)

            #Non-dominated sorting to identify the non-dominated solutions
            #on the current population
            self.ndsort.fast_nondominated_sort(self.population)
            for individual in self.population.fronts[0]:
                #avoids replacing the same individual in the external population
                if tuple(individual.features) not in self.visited_external:
                    self.visited_external.add(tuple(individual.features))
                    self.external_population.append(individual)
            lst = []
            lst_x = []
            lst_y = []

            #Non-dominated sorting to identify the non_dominated solutions
            #of the external population
            self.ndsort.fast_nondominated_sort(self.external_population)
            for individual in self.external_population.fronts[0]:
                lst.append(individual.objectives)
                lst_x.append(individual.objectives[0])
                lst_y.append(individual.objectives[1])

            #Plot the non-dominated solutions found so far during the optimization
            plt.scatter(lst_x, lst_y, marker='o', color='#0139DD', s=17)
            plt.title("Non-dominated solutions found so far")
            plt.xlabel('f1 (max)')
            plt.ylabel('f2 (max)')
            plt.show()
            plt.draw()
            plt.pause(0.003)
            plt.clf()

        plt.close()


