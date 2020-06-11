import numpy as np
import networkx as nx
from numba import njit


class MutationStrategies:
    def __init__(self, n_searches: int = 5):
        self.n_searches = n_searches

    def fitness_function(self, individual):
        pass

    def local_search(self, route, method, **kwargs):

        """
        It applies a 2-opt local search to a particular route. It selects 10 combinations
        of nodes on a given tour, performs a 2-opt node flipping,
        calculates the fitness of the resulting new tours and then selects the one
        with maximum fitness

        :param route: a given route to be mutated.
        :param method: method to use to perform local search.
        :return: the mutated route
        """

        mutation_size = kwargs["n_elements"] if "n_elements" in kwargs else 2

        mutation_cols = self.get_consecutive_mutation_cols(
            route, self.n_searches, mutation_size
        )

        local_search_rows = np.array(
            list(
                map(
                    lambda args: method(*args),
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

        route_fitness = self.fitness_function(route)

        max_fitness = np.argmax(fitness)

        return (
            local_search_rows[max_fitness, :]
            if fitness[max_fitness] > route_fitness
            else route
        )

    def two_opt_mutation(self, population, mutation_rows):

        population[mutation_rows, :] = np.array(
            list(
                map(
                    lambda route: self.two_opt_mutation_helper(route),
                    population[mutation_rows, :],
                )
            )
        )

        return population

    def two_opt_mutation_helper(self, route):

        mutation_cols = self.get_consecutive_mutation_cols(route, 2, 2, replace=False)

        rolled_arr = np.roll(route, route.shape[0] - mutation_cols[1, 1])

        next_index = np.argwhere(rolled_arr == route[mutation_cols[0, 0]])[0, 0]

        return np.hstack(
            (rolled_arr[: next_index + 1], np.flip(rolled_arr[next_index + 1 :]))
        )

    def random_inversion_mutation(self, population, mutation_rows, n_elements):
        """
        Performs a series of random flips and returns route with
        best fitness

        :param population: the current population
        :param mutation_rows: the indexes of the rows to be mutated
        :param n_elements: number of elements to be flipped
        :return: the mutated population
        """

        population[mutation_rows, :] = np.array(
            list(
                map(
                    lambda route: self.local_search(
                        route,
                        self.random_inversion_mutation_helper,
                        **{"n_elements": n_elements}
                    ),
                    population[mutation_rows, :],
                )
            )
        )

        return population

    @staticmethod
    def random_inversion_mutation_helper(route, mutation_cols):
        """
        Flips the nodes between 2 node positions for a given route.

        :param route: a given route to be mutated
        :param mutation_cols: the start position of the nodes to be flipped in between
        :return: the mutated route
        """

        np.put(route, mutation_cols, np.flip(route[mutation_cols]), mode="wrap")

        return route

    def random_gene_nearest_neighbour_mutation(self, population, mutation_rows):
        """
        Random gene inserted beside nearest neighbour mutation (RGIBNNM)

        4.2.7 - https://arxiv.org/pdf/1801.02827.pdf

        :param population: the current population
        :param mutation_rows: the indexes of the rows to be mutated
        :return: the mutated population
        """

        population[mutation_rows, :] = np.array(
            list(
                map(
                    lambda route: self.random_gene_nearest_neighbour_helper(route,),
                    population[mutation_rows, :],
                )
            )
        )

        return population

    def random_gene_nearest_neighbour_helper(self, route):
        """
        Random gene inserted beside nearest neighbour mutation (RGIBNNM)
        helper function

        :param route: current route to be mutated
        :return: the mutated route
        """

        random_gene = np.random.choice(route, 1)[0]

        random_gene_index = np.argwhere(route == random_gene)[0, 0]

        closest_neighbour = list(nx.neighbors(self.G, random_gene))[0]

        closest_neighbour_index = np.argwhere(route == closest_neighbour)[0, 0]

        if random_gene_index < closest_neighbour_index:
            return np.hstack(
                (
                    route[:random_gene_index],
                    route[random_gene_index + 1 : closest_neighbour_index + 1],
                    route[random_gene_index],
                    route[closest_neighbour_index + 1 :],
                )
            )

        else:
            return np.hstack(
                (
                    route[:closest_neighbour_index],
                    route[random_gene_index],
                    route[closest_neighbour_index:random_gene_index],
                    route[random_gene_index + 1 :],
                )
            )

    @staticmethod
    def random_swap_mutation(population, mutation_rows, mutation_cols):
        """
        Performs a swap between elements along rows

        :param population: the current population
        :param mutation_rows: the rows to be mutated
        :param mutation_cols: an matrix containing the indexes of the
        items to be swapped along the rows
        :return: the mutated population
        """

        (
            population[mutation_rows, mutation_cols[:, 0]],
            population[mutation_rows, mutation_cols[:, 1]],
        ) = (
            population[mutation_rows, mutation_cols[:, 1]],
            population[mutation_rows, mutation_cols[:, 0]],
        )

        return population

    @staticmethod
    def swap_genes(route, mutation_cols):
        """
        Performs a swap between elements along rows

        :param population: the current population
        :param mutation_rows: the rows to be mutated
        :param mutation_cols: an matrix containing the indexes of the
        items to be swapped along the rows
        :return: the mutated population
        """

        (route[mutation_cols[0]], route[mutation_cols[1]],) = (
            route[mutation_cols[1]],
            route[mutation_cols[0]],
        )

        return route

    def worst_gene_random_mutation(self, population, mutation_rows):

        """
        Worst left right gene with random gene mutation (WLRGWRGM)

        4.2.3 - https://arxiv.org/pdf/1801.02827.pdf

        :param population: the current population
        :param mutation_rows: the indexes of the rows to be mutated
        :return: the mutated population
        """

        population[mutation_rows, :] = np.array(
            list(
                map(
                    lambda route: self.worst_gene_random_mutation_helper(route,),
                    population[mutation_rows, :],
                )
            )
        )

        return population

    def worst_gene_random_mutation_helper(self, route):
        """
        Worst left right gene with random gene mutation (WLRGWRGM) helper function

        :param route: current route to be mutated
        :return: the mutated route
        """

        worst_gene = self.find_worst_gene(route)

        random_gene = worst_gene
        while random_gene == worst_gene:
            random_gene = np.random.choice(np.arange(route.shape[0]), 1)[0]

        return self.swap_genes(route, np.array([random_gene, worst_gene]))

    def worst_gene_nearest_neighbour_mutation(self, population, mutation_rows):
        """
        Worst gene with nearest neighbour mutation (WGWNNM)

        4.2.4 - https://arxiv.org/pdf/1801.02827.pdf

        :param population: the current population
        :param mutation_rows: the indexes of the rows to be mutated
        :return: the mutated population
        """

        population[mutation_rows, :] = np.array(
            list(
                map(
                    lambda route: self.worst_gene_random_mutation_helper(route,),
                    population[mutation_rows, :],
                )
            )
        )

        return population

    def worst_gene_nearest_neighbour_mutation_helper(self, route):
        """
        Worst gene with nearest neighbour mutation (WGWNNM) helper function

        :param route: current route to be mutated
        :return: the mutated route
        """

        worst_gene = self.find_worst_gene(route)

        closest_neighbour = list(nx.neighbors(self.G, route[worst_gene]))[0]

        closest_neighbour_index = np.argwhere(route == closest_neighbour)[0, 0]

        gene_candidates = np.take(
            np.arange(route.shape[0]),
            np.arange(closest_neighbour_index - 5, closest_neighbour_index + 5 + 1),
            mode="wrap",
        )

        selected_gene = np.random.choice(gene_candidates, 1)[0]

        return self.swap_genes(route, np.array([selected_gene, worst_gene]))

    @staticmethod
    def get_mutation_rows(n_mutations, population):
        """
        Retrieves the mutation rows for the current iteration

        :param n_mutations: number of mutations to be performed
        :param population: the current population
        :return: the mutation rows
        """

        mutation_rows = np.random.choice(
            np.arange(1, population.shape[0]), n_mutations, replace=False
        )

        return mutation_rows

    @staticmethod
    def get_mutation_cols(n_mutations, population):
        """
        Retrieves the mutation columns for the current iteration

        :param n_mutations: number of mutations to be performed
        :param population: the current population
        :return: the mutation columns
        """

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

        return mutation_cols

    @staticmethod
    def get_consecutive_mutation_cols(route, n_rows, slice_size, replace=True):

        return np.apply_along_axis(
            lambda gene_index: np.take(
                np.arange(route.shape[0]),
                np.arange(gene_index, gene_index + slice_size),
                mode="wrap",
            ),
            1,
            np.random.choice(route.shape[0], size=(n_rows, 1), replace=replace),
        )
