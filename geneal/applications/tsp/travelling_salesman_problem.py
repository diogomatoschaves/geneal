from functools import reduce

import numpy as np
import networkx as nx

from geneal.genetic_algorithms import ContinuousGenAlgSolver


class TravellingSalesmanProblemSolver(ContinuousGenAlgSolver):
    def __init__(self, graph, *args, **kwargs):

        if "n_crossover_points" in kwargs:
            if kwargs["n_crossover_points"] != 2:
                print("Defaulting 'n_crossover_points' to 2")
            kwargs.pop("n_crossover_points")

        ContinuousGenAlgSolver.__init__(self, n_crossover_points=2, *args, **kwargs)

        self.G = graph

    def fitness_function(self, individual):
        """
        Implements the logic that calculates the fitness
        measure of an individual. It sums all the costs of going
        from node to node in the tour.

        :param individual: chromosome of genes representing an individual
        :return: the fitness of the individual
        """

        res = reduce(
            lambda total_length, city_pair: total_length
            + self.G.edges[(city_pair[0], city_pair[1])]["weight"],
            zip(individual, individual[1:]),
            0,
        )

        res += self.G.edges[(241, individual[0])]["weight"]
        res += self.G.edges[(individual[-1], 241)]["weight"]

        return -round(res, 2)

    def initialize_population(self, pop_size, n_genes):
        """
        Initializes the population of the problem. It creates a
        matrix of size (pop_size x n_genes) containing permutations of the nodes
        on each row.

        :param pop_size: number of individuals in the population
        :param n_genes: number of genes representing the problem. In case of the binary
        solver, it represents the number of genes times the number of bits per gene
        :return: a numpy array with a randomized initialized population
        """

        population = np.repeat(
            np.arange(1, n_genes + 1)[np.newaxis, :], pop_size, axis=0
        )

        return np.array(list(map(lambda x: np.random.permutation(x), population)))

    def create_offspring(self, first_parent, sec_parent, crossover_pt, _):
        """
        Creates an offspring from 2 parents. It performs an OX crossover, which
        combines genes from each parent, but maintaining the nodes order of the parents.

        http://www.inf.tu-dresden.de/content/institutes/ki/cl/study/summer14/pssai/slides/GA_for_TSP.pdf

        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: points at which to perform the crossover
        :return: the resulting offspring.
        """

        reordered_sec_parent = np.roll(sec_parent, -crossover_pt[1])

        new_arr = first_parent[crossover_pt[0] : crossover_pt[1]]

        new_arr = np.append(new_arr, reordered_sec_parent)

        _, idx = np.unique(new_arr, return_index=True)

        res = np.roll(new_arr[np.sort(idx)], crossover_pt[0])

        if res.shape[0] != 30:
            a = 1

        return res

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population using a 2-opt rule hybrid. It selects the number of rows
        on which mutation will be applied, and then a applies a local search 2-opt rule
        to those rows.

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed. This number is
        calculated according to mutation_rate, but can be adjusted as needed inside this function
        :return: the mutated population
        """

        adjusted_n_mutations = np.ceil(n_mutations / self.n_genes).astype(int)

        mutation_rows, mutation_cols = self.get_mut_rows_cols(
            adjusted_n_mutations, population
        )

        population[mutation_rows, :] = np.array(
            list(
                map(
                    lambda route: self.local_search(route),
                    population[mutation_rows, :],
                )
            )
        )

        return population

    def local_search(self, route):

        """
        It applies a 2-opt local search to a particular route. It selects 10 combinations
        of nodes on a given tour, performs a 2-opt node flipping,
        calculates the fitness of the resulting new tours and then selects the one
        with maximum fitness

        :param route: a given route to be mutated.
        :return: the mutated route
        """

        mutation_cols = np.sort(np.random.choice(route.shape[0], size=(10, 2)), axis=1)

        local_search_rows = np.array(
            list(
                map(
                    lambda args: self.two_opt_swap(*args),
                    zip(
                        np.repeat(route[np.newaxis, :], mutation_cols.shape[0], axis=0),
                        mutation_cols,
                    ),
                )
            )
        )

        fitness = np.array(
            list(map(lambda row: self.fitness_function(row), local_search_rows))
        )
        max_fitness = np.argmax(fitness)

        return local_search_rows[max_fitness, :]

    @staticmethod
    def two_opt_swap(route, mutation_cols):
        """
        Flips the nodes between 2 node positions for a given route.

        :param route: a given route to be mutated
        :param mutation_cols: the position of the nodes to be flipped in between
        :return: the mutated route
        """
        route[mutation_cols[0] : mutation_cols[1]] = np.flip(
            route[mutation_cols[0] : mutation_cols[1]]
        )

        return route

    @staticmethod
    def get_mut_rows_cols(n_mutations, population):

        mutation_rows = np.random.choice(
            np.arange(1, population.shape[0]), n_mutations, replace=False
        )

        # mutation_cols = np.sort(np.random.choice(population.shape[1], size=(adjusted_n_mutations, 2)), axis=1)

        mutation_cols = np.array(
            list(
                map(
                    lambda x: np.sort(
                        np.random.choice(
                            np.arange(population.shape[1]), 2, replace=False
                        )
                    ),
                    range(n_mutations),
                )
            )
        ).astype(int)

        return mutation_rows, mutation_cols
