class Population:
    """
    Class that holds the entire population
    """

    def __init__(self):
        """
        Class constructor
        """
        self.population = []
        self.last_id = 0
        self.var_count = []
        self.probs = []
        self.probs_cluster = []
        self.fronts = []
        self.distances = []

    def __len__(self):
        """
        Returns the population length
        :return:
        """
        return len(self.population)

    def __iter__(self):
        """
        Allows the iteration through the class objects
        :return:
        """
        return self.population.__iter__()

    def extend(self, new_individuals):
        """
        Extends the current population with a set of new individuals
        :param new_individuals: set of new individuals
        :return:
        """
        self.population.extend(new_individuals)

    def append(self, new_individual):
        """
        Add another individual to the current population
        :param new_individual: new individual solution
        :return:
        """
        self.population.append(new_individual)