import numpy as np

from geneal.genetic_algorithms._genetic_algorithm import GenAlgSolver
from geneal.utils.helpers import get_input_dimensions


class ContinuousGenAlgSolver(GenAlgSolver):
    def __init__(
            self,
            fitness_function,
            n_genes: int,
            max_gen: int = 1000,
            pop_size: int = 100,
            mutation_rate: float = 0.15,
            selection: float = 0.5,
            variables_limits=None,
            problem_type=float
    ):
        """
        :param fitness_function:
        :param n_genes:
        :param max_gen:
        :param pop_size:
        :param mutation_rate:
        :param selection:
        """

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            n_genes=n_genes,
            max_gen=max_gen,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            selection=selection
        )

        if not variables_limits:
            min_max = np.iinfo(np.int64)
            variables_limits = [(min_max.min, min_max.max) for _ in range(n_genes)]

        if get_input_dimensions(variables_limits) == 1:
            variables_limits = [variables_limits for _ in range(n_genes)]

        self.variables_limits = variables_limits
        self.problem_type = problem_type

    def initialize_population(self):

        population = np.empty(shape=(self.pop_size, self.n_genes))

        for i, variable_limits in enumerate(self.variables_limits):
            if self.problem_type == float:
                population[:, i] = np.random.uniform(variable_limits[0], variable_limits[1], size=self.pop_size)
            else:
                population[:, i] = np.random.randint(variable_limits[0], variable_limits[1] + 1, size=self.pop_size)

        return population

    def create_offspring(self, first_parent, sec_parent, crossover_pt, offspring_number):

        beta = np.random.rand(1)[0] if offspring_number == "first" else -np.random.rand(1)[0]

        if self.problem_type == float:
            p_new = first_parent[crossover_pt] - beta * (first_parent[crossover_pt] - sec_parent[crossover_pt])
        else:
            p_new = first_parent[crossover_pt] - np.ceil(beta * (first_parent[crossover_pt] - sec_parent[crossover_pt]))

        return np.hstack((first_parent[:crossover_pt], p_new, sec_parent[crossover_pt+1:]))

    def mutate_variables(self, population):
        return self.initialize_population()
