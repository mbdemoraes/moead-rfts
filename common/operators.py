import random



class Operators():
    """
    Class that contains the genetic operators functions
    """

    def __init__(self,
                 problem):
        """
        Class constructor
        :param problem: object of the Problem class
        """
        self.problem = problem

    def crossover_binary(self, individual1, individual2, population):
        """
        Performs a two-point crossover
        :param individual1: parent solution 1
        :param individual2: parent solution 2
        :param population: current population
        :return: an offspring solution made by the crossover operation
        """
        population.last_id += 1
        child1 = self.problem.generate_individual()
        child1.id = population.last_id


        geneA = int(random.random() * len(individual1.features) - 2)
        geneB = int(random.random() * len(individual1.features) - 2)

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(0, startGene):
            child1.features[i] = individual1.features[i]

        for i in range(startGene, endGene):
            child1.features[i] = individual2.features[i]

        for i in range(endGene, len(individual1.features)):
            child1.features[i] = individual1.features[i]

        return child1


    def mutate_binary(self, child):
        """
        Mutate a child solution
        :param child: a non-mutated offspring solution
        :return: a mutated offspring
        """
        num_of_features = len(child.features)
        for gene in range(num_of_features):
            u = random.uniform(0, 1)
            prob = self.problem.mutation
            if u < prob:
                if child.features[gene] == 1:
                    child.features[gene] = 0
                else:
                    child.features[gene] = 1


