import numpy as np
from common.population import Population
import copy
import random
from sklearn.ensemble import RandomForestRegressor

class RANDOM_FORESTS:
    """
    Class that controls the Random Forest predictions and training.
    """

    def __init__(self, problem,
                 n_estimators,
                 criterion,
                 max_depth,
                 max_features,
                 min_samples_leaf,
                 min_samples_split,
                 max_samples):
        """
        Constructor of the problem
        :param problem: object of the class problem
        :param n_estimators: number of estimators (trees)
        :param criterion: criterion that defines how to measure the quality of a split
        :param max_depth: maximum depth of a tree
        :param max_features: number of decision variables to consider when building each tree
        :param min_samples_leaf: minimum number of samples to be at leaf node
        :param min_samples_split: minimum number of samples to split a node
        :param max_samples: number of decision vectors (samples) to consider when building each tree
        """
        self.problem = problem
        self.mean_error = np.zeros(self.problem.num_of_generations)
        self.random_forest_regressor = RandomForestRegressor(n_estimators=n_estimators,
                                                             max_depth=max_depth,
                                                             max_features=max_features,
                                                             min_samples_leaf=min_samples_leaf,
                                                             min_samples_split=min_samples_split,
                                                             criterion=criterion,
                                                             max_samples=max_samples,
                                                             bootstrap=True,
                                                             n_jobs=-1)
        self.train_vectors = []
        self.train_obj_fuctions = []


    def get_center_vector(self, individual_number, population, neighborhoods, neighborhood_size):
        """
        Calculates the center vector of a given neighborhood B
        :param individual_number: used to identify which neighborhood the individual belongs
        :param population: current population
        :param neighborhoods: neighborhoods
        :param neighborhood_size: neighborhood size
        :return: the center vector of the neighborhood B
        """
        center_vector = [0] * self.problem.num_of_objectives
        for k in range(neighborhood_size):
            individual = population.population[neighborhoods[individual_number][k]]
            for l in range(len(individual.objectives)):
                center_vector[l] += individual.objectives[l]
        center_vector = [(i/neighborhood_size) for i in center_vector]
        return center_vector

    def local_search_bin_mucop(self, child):
        """
        Performs a local search on a given child
        by randomly removing an item and randomly including another
        :param child: the offspring solution
        :return: the modified child solution
        """
        items_to_remove = [i for i in range(len(child.features)) if child.features[i] == 1]
        items_to_add = [i for i in range(len(child.features)) if child.features[i] == 0]
        copy_child = copy.deepcopy(child)
        choice_remove = random.choice(items_to_remove)
        choice_add = random.choice(items_to_add)
        copy_child.features[choice_remove] = 0
        copy_child.features[choice_add] = 1
        items_to_remove.remove(choice_remove)
        items_to_add.remove(choice_add)
        child.features = copy_child.features
        return copy_child


    def local_search_rf(self, child, individual_number, population, neighborhoods, neighborhood_size):
        """
        Function that modifies a given offspring solution to avoid generating
        duplicated solutions
        :param child: an offspring solution
        :param individual_number: used to identify which neighborhood the individual belongs
        :param population: current population
        :param neighborhoods: neighborhoods
        :param neighborhood_size: neighborhood size
        :return: a modified offspring solution with the closest estimation to the neighborhood
        center vector
        """
        children = Population()
        set_forbidden = set()
        set_forbidden.add(tuple(child.features))
        childs_features = []
        center_vector = self.get_center_vector(individual_number, population, neighborhoods, neighborhood_size)
        distances = []
        for k in range(self.problem.num_of_individuals):
            copy_child = copy.deepcopy(child)
            # If child is in the tabu list or have already been created in this loop
            # create another
            while tuple(copy_child.features) in self.problem.tabu or tuple(copy_child.features) in set_forbidden:
                copy_child = self.local_search_bin_mucop(copy_child)

            children.append(copy_child)
            childs_features.append(copy_child.features)
            set_forbidden.add(tuple(copy_child.features))

        # predict the objective function values of each child
        predictions = self.random_forest_regressor.predict(np.array(childs_features))  # Calculate the absolute errors
        for pred, calc_child in zip(predictions, children):
            calc_child.objectives = [int(i) for i in pred]

        # Get the individual with the closest estimation to the center vector
        for calc_child in children:
            distance = np.linalg.norm(np.array(calc_child.objectives) - np.array(center_vector))
            distances.append(distance)

        index_min = min(range(len(distances)), key=distances.__getitem__)
        child.features = children.population[index_min].features
        child.came_from = True

    def fit_population(self, train_features, train_labels):
        """
        Train the Random Forest with the initial population data
        :param train_features: initial population decision vectors
        :param train_labels: initial population objective function values
        :return:
        """
        self.train_vectors = copy.deepcopy(train_features)
        self.train_obj_fuctions = copy.deepcopy(train_labels)
        self.random_forest_regressor.fit(np.array(self.train_vectors), np.array(self.train_obj_fuctions))


    def predict_and_train(self, children_vectors, children_obj_function):
        """
        Function that first predicts the values of a list of children solutions
        Then use their data to retrain the Random Forest
        The children data is combined with the already known solutions
        :param children_vectors: children decision vectors
        :param children_obj_function: children objective function values
        :return:
        """
        predictions = self.random_forest_regressor.predict(np.array(children_vectors))
        self.train_vectors = np.vstack((self.train_vectors, children_vectors))
        self.train_obj_fuctions = np.vstack((self.train_obj_fuctions, children_obj_function))
        self.random_forest_regressor.fit(np.array(self.train_vectors), np.array(self.train_obj_fuctions))
        errors = abs(predictions - np.array(children_obj_function))  # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
