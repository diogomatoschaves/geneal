import datetime
import math
import inspect
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

from geneal.utils.helpers import print_elapsed_time


class GenAlgSolver(metaclass=ABCMeta):

    def __init__(
        self,
        fitness_function,
        n_genes: int,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
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
        """

        self.n_genes = n_genes
        self.max_gen = max_gen
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.selection = selection_rate

        self.pop_keep = math.floor(selection_rate * pop_size)

        self.generations_ = 0
        self.best_fitness_ = 0
        self.best_individual_ = None

        self.fitness_function = fitness_function
        self.initialize_population = self.initialize_population
        self.create_offspring = self.create_offspring
        self.mutate_population = self.mutate_population

        self.handle_overriding(fitness_function)

    def handle_overriding(self, fitness_function):
        if inspect.isclass(fitness_function):

            overriding_class = fitness_function()

            try:
                self.fitness_function = overriding_class.fitness_function
            except AttributeError:
                raise Exception("Overriding class must implement a 'fitness_function' method")

            try:
                self.initialize_population = overriding_class.initialize_population
            except AttributeError:
                pass

            try:
                self.create_offspring = overriding_class.create_offspring
            except AttributeError:
                pass

            try:
                self.mutate_population = overriding_class.mutate_population
            except AttributeError:
                pass

    def solve(self):

        start_time = datetime.datetime.now()

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

        find_parent = np.vectorize(
            lambda value: np.argmin(value > prob_intervals) - 1
        )

        number_matings = math.floor((self.pop_size - self.pop_keep) / 2)

        gen_n = 0
        while True:

            gen_n += 1

            print(gen_n)

            mean_fitness = np.append(mean_fitness, fitness.mean())
            max_fitness = np.append(max_fitness, fitness[0])

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

            mask = self.mutate_population(population)

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

        end_time = datetime.datetime.now()

        time_str = print_elapsed_time(start_time, end_time)

        self.print_stats(time_str)

    def calculate_fitness(self, population):
        return np.array(list(map(self.fitness_function, population)))

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

    def print_stats(self, time_str):

        print("\n#############################")
        print("#\t\t\tSTATS\t\t\t#")
        print("#############################\n\n")
        print(f"Total running time: {time_str}\n\n")
        print(f"Population size: {self.pop_size}")
        print(f"Number variables: {self.n_genes}")
        print(f"Selection rate: {self.selection}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Number Generations: {self.generations_}\n")
        print(f"Best fitness: {self.best_fitness_}")
        print(f"Best individual: {self.best_individual_}")

    @abstractmethod
    def initialize_population(self):
        pass

    @abstractmethod
    def create_offspring(self, first_parent, sec_parent, crossover_pt, offspring_number):
        pass

    @abstractmethod
    def mutate_population(self, population):
        pass