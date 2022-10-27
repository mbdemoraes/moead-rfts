class Individual(object):
    """
    Class that contains the informations of an individual solution
    """

    def __init__(self, directions):
        """
        Class constructor
        :param directions: the problem directions (maximization or minimization) for each objective function
        """
        self.id = 0
        self.rank = None
        self.domination_count = None
        self.num_dominated = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None
        self.came_from = None
        self.directions = directions

    def dominates(self, other_individual):
        """
        Verify if the individual dominates another individual
        :param other_individual: another individual solution
        :return: True if it dominates the other individual, False otherwise
        """
        conditions = []
        final_condition = False
        for k in range(len(self.directions)):
            if self.directions[k] == "max":
                if self.objectives[k] >= other_individual.objectives[k]:
                    conditions.append(True)
                else:
                    conditions.append(False)
            else:
                if self.objectives[k] <= other_individual.objectives[k]:
                    conditions.append(True)
                else:
                    conditions.append(False)

        if False not in conditions:
            for k in range(len(self.directions)):
                if self.directions[k] == "max":
                    if self.objectives[k] > other_individual.objectives[k]:
                        final_condition = True
                        break
                else:
                    if self.objectives[k] < other_individual.objectives[k]:
                        final_condition = True
                        break
        else:
            final_condition = False

        return final_condition











