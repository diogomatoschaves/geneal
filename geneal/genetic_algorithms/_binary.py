import math
import numpy as np

from geneal.genetic_algorithms._genetic_algorithm import GenAlgSolver


class BinaryGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        fitness_function,
        n_genes: int,
        n_bits: int = 1,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection: float = 0.5,
    ):
        """
        :param fitness_function:
        :param n_genes:
        :param n_bits:
        :param max_gen:
        :param pop_size:
        :param mutation_rate:
        :param selection:
        """

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            n_genes=n_genes * n_bits,
            max_gen=max_gen,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            selection=selection
        )

    def initialize_population(self):
        return np.round(np.random.rand(self.pop_size, self.n_genes))

    def create_offspring(self, first_parent, sec_parent, crossover_pt, offspring_number):
        return np.hstack((first_parent[:crossover_pt], sec_parent[crossover_pt:]))

    def mutate_variables(self, population):
        return np.abs(population - 1)
