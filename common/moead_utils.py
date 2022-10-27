import numpy as np
from itertools import  combinations

class MoeadUtils:
    """
    Class that contains useful functions for MOEA/D based methods
    """

    def __init__(self,
                 problem):
        """
        Class constructor
        :param problem: object of the Problem class
        """

        self.problem = problem


    def factorial(self, n):
        """
        Calculate the factorial of a number N
        :param n: the number to be used on the factorial calculation
        :return: the factorial result
        """
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result


    def comb(self, n, k):
        """
        Calculate the number of combinations of a sample of K elements from a set of N distinct objects
        where order does not matter and replacements are not allowed.
        :param n: number of distinct objects
        :param k: sample of elements
        :return: the possible number of combinations
        """
        return self.factorial(n) / (self.factorial(n - k) * self.factorial(k))

    def simplex_lattice_design(self, N, M):
        """
        Performs the Simplex Lattice Design to generate a set of uniform spread weight vectors
        :param N: number of desired individuals
        :param M: number of objective functions
        :return: the set of uniform spread weight vectors
        """
        try:
            H1 = 1
            while (self.comb(H1 + M, M - 1) <= N):
                H1 = H1 + 1

            temp1 = list(combinations(np.arange(H1 + M - 1), M - 1))
            temp1 = np.array(temp1)
            temp2 = np.arange(M - 1)
            temp2 = np.tile(temp2, (int(self.comb(H1 + M - 1, M - 1)), 1))
            W = temp1 - temp2
            W = (np.concatenate((W, np.zeros((np.size(W, 0), 1)) + H1), axis=1) - np.concatenate(
                (np.zeros((np.size(W, 0), 1)), W), axis=1)) / H1

            if H1 < M:
                H2 = 0
                while (self.comb(H1 + M - 1, M - 1) + self.comb(H2 + M, M - 1) <= N):
                    H2 = H2 + 1
                if H2 > 0:
                    temp1 = list(combinations(np.arange(H2 + M - 1), M - 1))
                    temp1 = np.array(temp1)
                    temp2 = np.arange(M - 1)
                    temp2 = np.tile(temp2, (int(self.comb(H2 + M - 1, M - 1)), 1))
                    W2 = temp1 - temp2
                    W2 = (np.concatenate((W2, np.zeros((np.size(W2, 0), 1)) + H2), axis=1) - np.concatenate(
                        (np.zeros((np.size(W2, 0), 1)), W2), axis=1)) / H2
                    W = np.concatenate((W, W2 / 2 + 1 / (2 * M)), axis=0)

            realN = np.size(W, 0)
            W[W == 0] = 10 ** (-6)
            if N!=realN:
                raise Exception("Population size unaivailable for the defined number of objectives.")
            return W, realN
        except Exception as error:
            print("Error while setting the weight vectors: " + repr(error))


    def set_neighborhoods(self, weights_vectors, num_of_individuals, number_of_obj_funcions, neighborhoood_size):
        """
        Set the neighboorhood by calculating the closest weight vectors to each weight vector
        :param weights_vectors: set of uniform spread weight vectors
        :param num_of_individuals: number of individuals in the population
        :param number_of_obj_funcions: number of objective functions in the problem
        :param neighborhoood_size: neighboorhood size
        :return: the neighborhood
        """
        neighboorhood = []
        for i in range(num_of_individuals):
            temp = []
            for j in range(num_of_individuals):
                distance = 0
                for k in range(number_of_obj_funcions):
                    distance += (weights_vectors[i][k] - weights_vectors[j][k]) ** 2
                distance = np.sqrt(distance)
                temp.append(distance)
            index = np.argsort(temp)
            neighboorhood.append(index[:neighborhoood_size])
        return neighboorhood


    def find_initial_reference_point(self, population):
        """
        Function that identifies the initial reference point
        considering the initial population data
        :param population: current population
        :return: the initial reference point
        """
        z = [np.inf for i in range(len(population.population[0].objectives))]
        for individual in population:
            for i in range(len(z)):
                if individual.objectives[0] < z[i]:
                    z[i] = individual.objectives[0]
        return z


    def update_reference_point(self, child, z):
        """
        Function that identifies if a child's objective functions
        have better objective functions than the current reference point
        :param child: an offspring solution
        :param z: current reference point
        :return: the updated reference point
        """

        for j in range(len(z)):
            if self.problem.directions[j] == "max":
                if child.objectives[j] > z[j]:
                    z[j] = child.objectives[j]
            else:
                if child.objectives[j] < z[j]:
                    z[j] = child.objectives[j]


    def Tchebycheff(self, individual, weight, z):
        """
        Performs the Tchebycheff decomposition
        :param individual: individual
        :param weight: weight vector
        :param z: current reference point
        :return: the Tchebycheff decomposition result
        """
        temp = []
        for i in range(len(individual.objectives)):
            temp.append(weight[i] * np.abs(individual.objectives[i] - z[i]))

        return np.max(temp)