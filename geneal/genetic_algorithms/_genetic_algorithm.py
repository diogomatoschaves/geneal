import math
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt


class GenAlgSolver(metaclass=ABCMeta):

    def __init__(
        self,
        fitness_function,
        n_genes: int,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection: float = 0.5,
    ):
        """
        :param fitness_function:
        :param n_genes:
        :param max_gen:
        :param pop_size:
        :param mutation_rate:
        :param selection:
        """

        self.fitness_function = fitness_function
        self.n_genes = n_genes
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.selection = selection

        self.pop_keep = math.floor(selection * pop_size)

        self.generations_ = 0
        self.best_fitness_ = 0
        self.best_individual_ = None

    def solve(self):

        mean_fitness = np.ndarray(shape=(1, 0))
        max_fitness = np.ndarray(shape=(1, 0))

        # initialize the population
        population = self.initialize_population()

        fitness = self.calculate_fitness(population)

        fitness, population = self.sort_by_fitness(fitness, population)

        mating_prob = (
            np.arange(1, self.pop_keep + 1) / np.arange(1, self.pop_keep + 1).sum()
        )[::-1]

        prob_intervals = np.array([0, *np.cumsum(mating_prob[: self.pop_keep + 1])])

        number_matings = math.floor((self.pop_size - self.pop_keep) / 2)

        gen_n = 0
        while True:

            gen_n += 1

            print(gen_n)

            mean_fitness = np.append(mean_fitness, fitness.mean())
            max_fitness = np.append(max_fitness, fitness[0])

            find_parent = np.vectorize(
                lambda value: np.argmin(value > prob_intervals) - 1
            )

            ma = find_parent(np.random.rand(number_matings))
            pa = find_parent(np.random.rand(number_matings))

            ix = np.arange(0, self.pop_size - self.pop_keep - 1, 2)  # index of mate #1

            # crossover point
            xp = np.round(np.random.rand(number_matings) * (self.n_genes - 1)).astype(
                int
            )

            for i in range(xp.shape[0]):

                # create first offspring
                population[self.pop_keep + ix[i], :] = self.create_offspring(
                    population[ma[i], :], population[pa[i], :], xp[i], "first"
                )

                # create second offspring
                population[self.pop_keep + ix[i] + 1, :] = self.create_offspring(
                    population[pa[i], :], population[ma[i], :], xp[i], "second"
                )

            # Mutate bits
            n_mutations = math.ceil(
                (self.pop_size - 1) * self.n_genes * self.mutation_rate
            )

            mutation_rows = np.ceil(
                np.random.rand(1, n_mutations) * (self.pop_size - 1)
            ).astype(int)

            mutation_cols = (
                    np.ceil(np.random.rand(1, n_mutations) * self.n_genes) - 1
            ).astype(int)

            mask = self.mutate_variables(population)

            population[mutation_rows, mutation_cols] = mask[
                mutation_rows, mutation_cols
            ]

            fitness = np.hstack((fitness[0], self.calculate_fitness(population[1:, :])))

            fitness, population = self.sort_by_fitness(fitness, population)

            if gen_n >= self.max_gen:
                break

        self.generations_ = gen_n
        self.best_fitness_ = fitness[0]
        self.best_individual_ = population[0, :]

        self.plot_results(mean_fitness, max_fitness, gen_n)

    def calculate_fitness(self, population):
        return np.apply_along_axis(self.fitness_function, 1, population)

    @staticmethod
    def sort_by_fitness(fitness, population):

        sorted_fitness = np.argsort(fitness)[::-1]

        population = population[sorted_fitness, :]
        fitness = fitness[sorted_fitness]

        return fitness, population

    @staticmethod
    def plot_results(mean_fitness, max_fitness, iterations):

        plt.figure(figsize=(7, 7))

        x = np.arange(1, iterations + 1)

        plt.plot(x, mean_fitness, label='mean fitness')
        plt.plot(x, max_fitness, label='max fitness')

        plt.legend()
        plt.show()

    @abstractmethod
    def initialize_population(self):
        pass

    @abstractmethod
    def create_offspring(self, first_parent, sec_parent, crossover_pt, offspring_number):
        pass

    @abstractmethod
    def mutate_variables(self, population):
        pass