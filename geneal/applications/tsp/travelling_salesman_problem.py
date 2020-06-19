import random
import time
from functools import reduce
import hashlib
from collections import defaultdict

import numpy as np
from numba import njit
from numba.core import types
from numba.typed import Dict

from geneal.applications.tsp.mutation_strategies import MutationStrategies
from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.utils.exceptions import InvalidInput


mutation_options = {"random_swap", "random_inversion", "2-opt"}

allowed_mutations = {
    "2-opt",
    "random_swap",
    "random_inversion",
    "random_gene_nearest_neighbour",
    "worst_gene_random",
    "worst_gene_nearest_neighbour",
    "select_any_mutation",
}


@njit
def fitness_function(individual, edges):
    """
    Implements the logic that calculates the fitness
    measure of an individual. It sums all the costs of going
    from node to node in the tour.

    :param individual: chromosome of genes representing an individual
    :param edges: dictionary with cost between all nodes
    :return: the fitness of the individual
    """

    total_length = 0
    for i in range(individual.shape[0] - 1):
        total_length += edges[(individual[i], individual[i + 1])]

    total_length += edges[(individual[0], individual[-1])]

    return -round(total_length, 2)


def convert_to_typed_dict(G):

    edges_dict = Dict.empty(
        key_type=types.UniTuple(types.int64, 2), value_type=types.float64
    )

    edges_dict.update({(edge[1], edge[0]): G.edges[edge]["weight"] for edge in G.edges})
    edges_dict.update({(edge[0], edge[1]): G.edges[edge]["weight"] for edge in G.edges})

    return edges_dict


class TravellingSalesmanProblemSolver(MutationStrategies, ContinuousGenAlgSolver):
    def __init__(
        self,
        graph,
        mutation_strategy: str = "2-opt",
        n_searches: int = 1,
        numba_speedup: bool = False,
        *args,
        **kwargs,
    ):

        self.check_input(kwargs, graph)

        MutationStrategies.__init__(self, n_searches=n_searches)
        ContinuousGenAlgSolver.__init__(self, n_crossover_points=2, *args, **kwargs)

        if mutation_strategy not in allowed_mutations:
            raise (InvalidInput(f"{mutation_strategy} is an invalid mutation strategy"))

        if numba_speedup:

            edges_dict = convert_to_typed_dict(graph)

            self.fitness_function = lambda individual: fitness_function(
                individual, edges_dict
            )

        self.G = graph
        self.mutation_strategy = mutation_strategy
        self.fitness_time = 0
        self.chromosomes = defaultdict(int)

    @staticmethod
    def check_input(kwargs, graph):
        if "n_crossover_points" in kwargs:

            if kwargs["n_crossover_points"] != 2:
                print("Defaulting 'n_crossover_points' to 2")
            kwargs.pop("n_crossover_points")

        if "n_genes" in kwargs:
            if kwargs["n_genes"] > len(graph.nodes):
                print(
                    f"'n_genes' can't be larger than the nodes in the graph. The number of genes "
                    f"will default to {len(graph.nodes)}."
                )

                kwargs["n_genes"] = len(graph.nodes)

        else:
            kwargs["n_genes"] = len(graph.nodes)

        return kwargs

    def fitness_function(self, individual):
        """
        Implements the logic that calculates the fitness
        measure of an individual. It sums all the costs of going
        from node to node in the tour.

        :param individual: chromosome of genes representing an individual
        :return: the fitness of the individual
        """

        start_time = time.time()

        arr_hash = hashlib.sha1(individual).hexdigest()

        if arr_hash in self.chromosomes:
            res = self.chromosomes[arr_hash]

        else:
            res = reduce(
                lambda total_length, city_pair: total_length
                + self.G.edges[(city_pair[0], city_pair[1])]["weight"],
                zip(individual, individual[1:]),
                0,
            )

            res += self.G.edges[(individual[0], individual[-1])]["weight"]

            res = -round(res, 2)

            self.chromosomes[arr_hash] = res

        self.fitness_time += time.time() - start_time

        return res

    def initialize_population(self):
        """
        Initializes the population of the problem. It creates a
        matrix of size (pop_size x n_genes) containing permutations of the nodes
        on each row.

        :return: a numpy array with a randomized initialized population
        """

        population = np.repeat(
            np.arange(1, self.n_genes + 1)[np.newaxis, :], self.pop_size, axis=0
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

    def mutate_population(self, population, n_mutations, **kwargs):
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

        if adjusted_n_mutations == 0:
            return population

        mutation_rows = self.get_mutation_rows(adjusted_n_mutations, population)

        mutation_strategy = self.mutation_strategy
        if "mutation_strategy" in kwargs:
            mutation_strategy = kwargs["mutation_strategy"]

        if mutation_strategy == "2-opt":

            return self.two_opt_mutation(population, mutation_rows)

        elif mutation_strategy == "random_swap":

            mutation_cols = self.get_mutation_cols(adjusted_n_mutations, population)

            return self.random_swap_mutation(population, mutation_rows, mutation_cols)

        elif mutation_strategy == "random_gene_around_nearest_neighbour":

            return self.random_gene_around_nearest_neighbour_mutation(
                population, mutation_rows
            )

        elif mutation_strategy == "random_gene_nearest_neighbour":

            return self.random_gene_nearest_neighbour_mutation(
                population, mutation_rows
            )

        elif mutation_strategy == "worst_gene_random":

            return self.worst_gene_random_mutation(population, mutation_rows)

        elif mutation_strategy == "worst_gene_nearest_neighbour":

            return self.worst_gene_nearest_neighbour_mutation(population, mutation_rows)

        elif mutation_strategy == "random_inversion":

            return self.random_inversion_mutation(
                population,
                mutation_rows,
                np.random.choice(int(population.shape[1] / 2), 1)[0],
            )

        elif mutation_strategy == "select_any_mutation":

            selected_strategy = random.sample(mutation_options, 1)[0]

            return self.mutate_population(
                population, n_mutations, **{"mutation_strategy": selected_strategy}
            )

        else:
            raise (InvalidInput(f"{mutation_strategy} is an invalid mutation strategy"))

    def find_worst_gene(self, chromosome):

        distances = [
            self.G.edges[(chromosome[-1], chromosome[0])]["weight"]
            + self.G.edges[(chromosome[0], chromosome[1])]["weight"],
            *[
                self.G.edges[(city_pair[0], city_pair[1])]["weight"]
                + self.G.edges[(city_pair[1], city_pair[2])]["weight"]
                for city_pair in zip(chromosome, chromosome[1:], chromosome[2:])
            ],
            self.G.edges[(chromosome[-2], chromosome[-1])]["weight"]
            + self.G.edges[(chromosome[-1], chromosome[0])]["weight"],
        ]

        worst_gene = np.argmax(distances)

        return worst_gene
