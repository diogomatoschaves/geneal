import numpy as np

from geneal.genetic_algorithms.genetic_algorithm_base import GenAlgSolver


class BinaryGenAlgSolver(GenAlgSolver):
    def __init__(
        self,
        n_genes: int,
        fitness_function=None,
        n_bits: int = 1,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        n_crossover_points: int = 1,
        random_state: int = None,
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
            selection_rate=selection_rate,
            n_crossover_points=n_crossover_points,
            random_state=random_state,
        )

    def initialize_population(self, pop_size, n_genes):
        """
        Initializes the population of the problem according to the
        population size and number of genes and outputs individuals
        composed of 0's and 1's.

        :param pop_size: number of individuals in the population
        :param n_genes: number of genes representing the problem, which corresponds to
        the number of genes times the number of bits per gene
        :return: a numpy array with a randomized initialized population
        """
        return np.round(np.random.rand(pop_size, n_genes))

    @staticmethod
    def create_offspring(first_parent, sec_parent, crossover_pt, _):
        """
        Creates an offspring from 2 parents. It performs the crossover
        according the following rule:

        offspring = [first_parent[:crossover_pt], sec_parent[crossover_pt:]
        """

        return np.hstack(
            (first_parent[: crossover_pt[0]], sec_parent[crossover_pt[0] :])
        )

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population by randomizing specific positions of the
        population individuals.

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :return: the mutated population
        """

        mutation_rows, mutation_cols = super(
            BinaryGenAlgSolver, self
        ).mutate_population(population, n_mutations)

        population[mutation_rows, mutation_cols] = np.abs(population - 1)[
            mutation_rows, mutation_cols
        ]

        return population
