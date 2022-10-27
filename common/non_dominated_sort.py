
class Non_Dominated_Sort:
    """
    Class that peforms fast non-dominated sorting to identify the non-dominated solutions
    """

    def fast_nondominated_sort(self, population):
        """
        Fast non-dominated sorting
        :param population: current population
        :return: population and the individuals at each front
        """
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.num_dominated = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
                    individual.num_dominated += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)
