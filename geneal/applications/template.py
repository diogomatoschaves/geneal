from geneal.genetic_algorithms import ContinuousGenAlgSolver


class TemplateChildClass(ContinuousGenAlgSolver):
    def __init__(self, *args, **kwargs):
        ContinuousGenAlgSolver.__init__(self, *args, **kwargs, n_crossover_points=2)

    def fitness_function(self):
        pass

    def initialize_population(self):
        pass

    def create_offspring(
        self, first_parent, sec_parent, crossover_pt, offspring_number
    ):
        pass

    def mutate_population(self, population, n_mutations):
        pass
