from common.individual import Individual
import random
from common.population import Population

class Problem:
    """
    Class that controls the problem and the optimization parameters such as
    number of individuals, number of generations, variables range, mutation rate,
    tabu list and the problem directions (maximization or minimization).
    """

    def __init__(self,
                 objectives,
                 num_of_variables,
                 variables_range,
                 num_of_individuals,
                 directions,
                 num_of_generations,
                 mutation,
                 expand=True):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.num_of_individuals = num_of_individuals
        self.objectives = objectives
        self.expand = expand
        self.variables_range = variables_range
        self.directions = directions
        self.num_of_generations = num_of_generations
        self.variables = self.set_variables()
        self.mutation = mutation
        self.tabu = set()

    def set_variables(self):
        """
        Set the possible variables values for each decision variable
        :return: The set of possible variables for the given problem
        """
        variables = [i for i in range(min(self.variables_range), max(self.variables_range) + 1)]
        return variables

    def create_update_tabu_list(self, population):
        """
        Create a tabu list (which is in fact a set) or update the current one
        :param population:
        :return: nothing
        """
        for individual in population:
            self.tabu.add(tuple(individual.features))

    def create_initial_population(self):
        """
        Create an initial population
        :return: return a population of N individuals
        """
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.generate_individual()
            individual.id = _
            individual.trace = [_ for i in range(self.num_of_variables)]
            self.calculate_objectives(individual)
            population.append(individual)
            population.last_id = _
        return population

    def generate_individual(self):
        """
        Generate an individual
        :return: an individual object
        """
        individual = Individual(self.directions)
        individual.features = [random.randint(min(self.variables_range), max(self.variables_range)) for x in range(self.num_of_variables)]
        return individual


    def calculate_objectives(self, individual):
        """
        Calculate the objective function values of an individual
        :param individual: solution containing the decision vector and the objective functions
        :return: nothing
        """
        if self.expand:
            individual.objectives = [f(*individual.features) for f in self.objectives]
        else:
            individual.objectives = [f(individual.features) for f in self.objectives]

