import numpy as np

from geneal.genetic_algorithms.genetic_algorithm_base import GenAlgSolver
from geneal.utils.helpers import get_input_dimensions


class ContinuousGenAlgSolver(GenAlgSolver):

    def __init__(
        self,
        n_genes: int,
        fitness_function=None,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        variables_limits=(-10, 10),
        problem_type=float,
        n_crossover_points: int = 1,
        random_state: int = None
    ):
        """
        :param fitness_function: can either be a fitness function or
        a class implementing a fitness function + methods to override
        the default ones: create_offspring, mutate_population, initialize_population
        :param n_genes: number of genes (variables) to have in each chromosome
        :param max_gen: maximum number of generations to perform the optimization
        :param pop_size: population size
        :param mutation_rate: rate at which random mutations occur
        :param selection_rate: percentage of the population to be selected for crossover
        :param variables_limits: limits for each variable [(x1_min, x1_max), (x2_min, x2_max), ...].
        If only one tuple is provided, then it is assumed the same for every variable
        :param problem_type: whether problem is of float or integer type
        """

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            n_genes=n_genes,
            max_gen=max_gen,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            selection_rate=selection_rate,
            n_crossover_points=n_crossover_points,
            random_state=random_state
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
                population[:, i] = np.random.uniform(
                    variable_limits[0], variable_limits[1], size=self.pop_size
                )
            else:
                population[:, i] = np.random.randint(
                    variable_limits[0], variable_limits[1] + 1, size=self.pop_size
                )

        return population

    def create_offspring(
        self, first_parent, sec_parent, crossover_pt, offspring_number
    ):

        crossover_pt = crossover_pt[0]

        beta = (
            np.random.rand(1)[0]
            if offspring_number == "first"
            else -np.random.rand(1)[0]
        )

        if self.problem_type == float:
            p_new = first_parent[crossover_pt] - beta * (
                first_parent[crossover_pt] - sec_parent[crossover_pt]
            )
        else:
            p_new = first_parent[crossover_pt] - np.round(
                beta * (first_parent[crossover_pt] - sec_parent[crossover_pt])
            )

        return np.hstack(
            (first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1 :])
        )

    def mutate_population(self, population, n_mutations):

        mutation_rows, mutation_cols = super(
            ContinuousGenAlgSolver, self
        ).mutate_population(population, n_mutations)

        population[mutation_rows, mutation_cols] = self.initialize_population()[
            mutation_rows, mutation_cols
        ]

        return population
