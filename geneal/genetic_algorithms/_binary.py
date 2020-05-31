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
        selection_rate: float = 0.5,
    ):
        """
        :param fitness_function: can either be a fitness function or a class implementing a fitness function +
        methods to override the default ones: create_offspring, mutate_population, initialize_population
        :param n_genes: number of genes (variables) to have in each chromosome
        :param n_bits: number of bits representing each gene
        :param max_gen: maximum number of generations to perform the optimization
        :param pop_size: population size
        :param mutation_rate: rate at which random mutations occur
        :param selection_rate: percentage of the population to be selected for crossover
        """

        GenAlgSolver.__init__(
            self,
            fitness_function=fitness_function,
            n_genes=n_genes * n_bits,
            max_gen=max_gen,
            pop_size=pop_size,
            mutation_rate=mutation_rate,
            selection_rate=selection_rate
        )

    def initialize_population(self):
        return np.round(np.random.rand(self.pop_size, self.n_genes))

    def create_offspring(self, first_parent, sec_parent, crossover_pt, offspring_number):
        return np.hstack((first_parent[:crossover_pt], sec_parent[crossover_pt:]))

    def mutate_population(self, population):
        return np.abs(population - 1)
