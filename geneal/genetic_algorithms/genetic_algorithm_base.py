import datetime
import logging
import math
from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from geneal.utils.exceptions import NoFitnessFunction, InvalidInput
from geneal.utils.exceptions_messages import exception_messages
from geneal.utils.helpers import get_elapsed_time
from geneal.utils.logger import configure_logger


allowed_selection_strategies = {"roulette_wheel", "two_by_two", "random", "tournament"}


class GenAlgSolver:
    def __init__(
        self,
        n_genes: int,
        fitness_function=None,
        max_gen: int = 1000,
        pop_size: int = 100,
        mutation_rate: float = 0.15,
        selection_rate: float = 0.5,
        selection_strategy: str = "roulette_wheel",
        verbose: bool = True,
        plot_results: bool = True,
        excluded_genes: Sequence = None,
        n_crossover_points: int = 1,
        random_state: int = None,
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
        :param selection_strategy: strategy to use for selection
        :param verbose: whether to print iterations status
        :param plot_results: whether to plot results of the run at the end
        :param n_crossover_points: number of slices to make for the crossover
        :param random_state: optional. whether the random seed should be set
        """

        if isinstance(random_state, int):
            np.random.seed(random_state)

        configure_logger()

        self.n_genes = n_genes
        self.allowed_mutation_genes = np.arange(self.n_genes)

        self.check_input_base(
            fitness_function, selection_strategy, pop_size, excluded_genes
        )

        self.selection_strategy = selection_strategy

        self.max_gen = max_gen
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.selection_rate = selection_rate
        self.n_crossover_points = n_crossover_points
        self.verbose = verbose
        self.plot_results = plot_results

        self.pop_keep = math.floor(selection_rate * pop_size)

        if self.pop_keep < 2:
            self.pop_keep = 2

        self.prob_intervals = self.get_selection_probabilities()

        self.n_matings = math.floor((self.pop_size - self.pop_keep) / 2)
        self.n_mutations = self.get_number_mutations()

        self.generations_ = 0
        self.best_fitness_ = 0
        self.best_individual_ = None
        self.population_ = None

    def check_input_base(
        self, fitness_function, selection_strategy, pop_size, excluded_genes
    ):

        if not fitness_function:
            try:
                getattr(self, "fitness_function")
            except AttributeError:
                raise NoFitnessFunction(
                    "A fitness function must be defined or provided as an argument"
                )
        else:
            self.fitness_function = fitness_function

        if selection_strategy not in allowed_selection_strategies:
            raise InvalidInput(
                exception_messages["InvalidSelectionStrategy"](
                    selection_strategy, allowed_selection_strategies
                )
            )

        if pop_size < 2:
            raise (InvalidInput(exception_messages["InvalidPopulationSize"]))

        if isinstance(excluded_genes, (list, tuple, np.ndarray)):
            self.allowed_mutation_genes = [
                item
                for item in self.allowed_mutation_genes
                if item not in excluded_genes
            ]

        elif excluded_genes is not None:
            raise InvalidInput(
                exception_messages["InvalidExcludedGenes"](excluded_genes)
            )

    def solve(self):
        """
        Performs the genetic algorithm optimization according to the parameters
        provided at initialization.

        :return: None
        """

        start_time = datetime.datetime.now()

        mean_fitness = np.ndarray(shape=(1, 0))
        max_fitness = np.ndarray(shape=(1, 0))

        # initialize the population
        population = self.initialize_population()

        fitness = self.calculate_fitness(population)

        fitness, population = self.sort_by_fitness(fitness, population)

        gen_interval = max(round(self.max_gen / 10), 1)

        gen_n = 0
        while True:

            gen_n += 1

            if self.verbose and gen_n % gen_interval == 0:
                logging.info(f"Iteration: {gen_n}")
                logging.info(f"Best fitness: {fitness[0]}")

            mean_fitness = np.append(mean_fitness, fitness.mean())
            max_fitness = np.append(max_fitness, fitness[0])

            ma, pa = self.select_parents(fitness)

            ix = np.arange(0, self.pop_size - self.pop_keep - 1, 2)

            xp = np.array(
                list(map(lambda _: self.get_crossover_points(), range(self.n_matings)))
            )

            for i in range(xp.shape[0]):

                # create first offspring
                population[-1 - ix[i], :] = self.create_offspring(
                    population[ma[i], :], population[pa[i], :], xp[i], "first"
                )

                # create second offspring
                population[-1 - ix[i] - 1, :] = self.create_offspring(
                    population[pa[i], :], population[ma[i], :], xp[i], "second"
                )

            population = self.mutate_population(population, self.n_mutations)

            fitness = np.hstack((fitness[0], self.calculate_fitness(population[1:, :])))

            fitness, population = self.sort_by_fitness(fitness, population)

            if gen_n >= self.max_gen:
                break

        self.generations_ = gen_n
        self.best_fitness_ = fitness[0]
        self.best_individual_ = population[0, :]
        self.population_ = population

        if self.plot_results:
            self.plot_fitness_results(mean_fitness, max_fitness, gen_n)

        if self.verbose:
            end_time = datetime.datetime.now()

            time_str = get_elapsed_time(start_time, end_time)

            self.print_stats(time_str)

    def calculate_fitness(self, population):
        """
        Calculates the fitness of the population

        :param population: population state at a given iteration
        :return: the fitness of the current population
        """
        return np.array(list(map(self.fitness_function, population)))

    def select_parents(self, fitness):
        """
        Selects the parents according to a given selection strategy.
        Options are

        :param fitness: the fitness values of the population at a given iteration
        :return: a tuple containing the selected 2 parents for each mating
        """

        ma, pa = None, None

        if self.selection_strategy == "roulette_wheel":

            ma = np.apply_along_axis(
                self.roulette_wheel_selection, 1, np.random.rand(self.n_matings, 1)
            )
            pa = np.apply_along_axis(
                self.roulette_wheel_selection, 1, np.random.rand(self.n_matings, 1)
            )

        elif self.selection_strategy == "two_by_two":

            range_max = int(self.n_matings * 2)

            ma = np.arange(range_max)[::2]
            pa = np.arange(range_max)[1::2]

            if ma.shape[0] > pa.shape[0]:
                ma = ma[:-1]

        elif self.selection_strategy == "random":

            ma = np.apply_along_axis(
                self.random_selection, 1, np.random.rand(self.n_matings, 1)
            )
            pa = np.apply_along_axis(
                self.random_selection, 1, np.random.rand(self.n_matings, 1)
            )

        elif self.selection_strategy == "tournament":

            range_max = int(self.n_matings * 2)

            ma = self.tournament_selection(fitness, range_max)
            pa = self.tournament_selection(fitness, range_max)

        return ma, pa

    def roulette_wheel_selection(self, value):
        return np.argmin(value > self.prob_intervals) - 1

    def random_selection(self, value):
        return np.argmin(value > self.prob_intervals) - 1

    def tournament_selection(self, fitness, range_max):

        selected_individuals = np.random.choice(range_max, size=(self.n_matings, 3))

        return np.array(
            list(
                map(
                    lambda x: self.tournament_selection_helper(x, fitness),
                    selected_individuals,
                )
            )
        )

    def tournament_selection_helper(self, selected_individuals, fitness):

        individuals_fitness = fitness[selected_individuals]

        return selected_individuals[np.argmax(individuals_fitness)]

    def get_selection_probabilities(self):

        if self.selection_strategy == "roulette_wheel":

            mating_prob = (
                np.arange(1, self.pop_keep + 1) / np.arange(1, self.pop_keep + 1).sum()
            )[::-1]

            return np.array([0, *np.cumsum(mating_prob[: self.pop_keep + 1])])

        elif self.selection_strategy == "random":
            return np.linspace(0, 1, self.pop_keep + 1)

    def get_number_mutations(self):
        return math.ceil(
            (self.pop_size - 1) * self.n_genes * self.mutation_rate
        )

    @staticmethod
    def sort_by_fitness(fitness, population):
        """
        Sorts the population by its fitness.

        :param fitness: fitness of the population
        :param population: population state at a given iteration
        :return: the sorted fitness array and sorted population array
        """

        sorted_fitness = np.argsort(fitness)[::-1]

        population = population[sorted_fitness, :]
        fitness = fitness[sorted_fitness]

        return fitness, population

    def get_crossover_points(self):
        """
        Retrieves random crossover points

        :return: a numpy array with the crossover points
        """
        return np.sort(
            np.random.choice(
                np.arange(self.n_genes + 1), self.n_crossover_points, replace=False
            )
        )

    @staticmethod
    def plot_fitness_results(mean_fitness, max_fitness, iterations):
        """
        Plots the evolution of the mean and max fitness of the population

        :param mean_fitness: mean fitness array for each generation
        :param max_fitness: max fitness array for each generation
        :param iterations: total number of generations
        :return: None
        """

        plt.figure(figsize=(7, 7))

        x = np.arange(1, iterations + 1)

        plt.plot(x, mean_fitness, label="mean fitness")
        plt.plot(x, max_fitness, label="max fitness")

        plt.legend()
        plt.show()

    def print_stats(self, time_str):
        """
        Prints the statistics of the optimization run

        :param time_str: time string given by the method get_elapsed_time
        :return: None
        """

        logging.info("\n#############################")
        logging.info("#\t\t\tSTATS\t\t\t#")
        logging.info("#############################\n\n")
        logging.info(f"Total running time: {time_str}\n\n")
        logging.info(f"Population size: {self.pop_size}")
        logging.info(f"Number variables: {self.n_genes}")
        logging.info(f"Selection rate: {self.selection_rate}")
        logging.info(f"Mutation rate: {self.mutation_rate}")
        logging.info(f"Number Generations: {self.generations_}\n")
        logging.info(f"Best fitness: {self.best_fitness_}")
        logging.info(f"Best individual: {self.best_individual_}")

    @abstractmethod
    def initialize_population(self):
        """
        Initializes the population of the problem. To be implemented in each child class.

        :return: a numpy array with a randomized initialized population
        """
        pass

    @staticmethod
    @abstractmethod
    def create_offspring(first_parent, sec_parent, crossover_pt, offspring_number):
        """
        Creates an offspring from 2 parents. It uses the crossover point(s)
        to determine how to perform the crossover. To be implemented on each child class.

        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: point(s) at which to perform the crossover
        :param offspring_number: whether it's the first or second offspring from a pair of parents.
        Important if there's different logic to be applied to each case.
        :return: the resulting offspring.
        """
        pass

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population according to a given user defined rule.
        To be defined further in each child class. Each direct child class can call
        this super method to retrieve the mutation rows and mutations columns

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed. This number is
        calculated according to mutation_rate, but can be adjusted as needed inside this function
        :return: an array with the mutation_rows and mutation_cols
        """

        mutation_rows = np.random.choice(
            np.arange(1, self.pop_size), n_mutations, replace=True
        )

        mutation_cols = np.random.choice(
            self.allowed_mutation_genes, n_mutations, replace=True
        )

        return [mutation_rows, mutation_cols]
