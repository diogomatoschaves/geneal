import time
from functools import reduce
import hashlib
from collections import defaultdict

import numpy as np
from numba import njit

from geneal.applications.tsp.mutation_strategies import MutationStrategies
from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.utils.exceptions import InvalidInput


mutation_options = {
    "random_swap",
    "random_inversion",
}

allowed_mutations = {
    "2-opt",
    "random_swap",
    "random_inversion",
    "random_gene_around_nearest_neighbour",
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
    :return: the fitness of the individual
    """

    total_length = 0
    for i in range(individual.shape[0] - 1):
        total_length += edges[(individual[i], individual[i + 1])]

    total_length += edges[(individual[0], individual[-1])]

    return -round(total_length, 2)


class TravellingSalesmanProblemSolver(MutationStrategies, ContinuousGenAlgSolver):
    def __init__(
        self,
        graph,
        mutation_strategy: str = "2-opt",
        n_searches: int = 5,
        *args,
        **kwargs,
    ):

        if "n_crossover_points" in kwargs:
            if kwargs["n_crossover_points"] != 2:
                print("Defaulting 'n_crossover_points' to 2")
            kwargs.pop("n_crossover_points")

        if "n_genes" in kwargs:
            print(
                f"'n_genes' is determined by the number of nodes in G ({len(graph.nodes)})"
            )
        kwargs["n_genes"] = len(graph.nodes)

        MutationStrategies.__init__(self, n_searches=n_searches)
        ContinuousGenAlgSolver.__init__(self, n_crossover_points=2, *args, **kwargs)

        if mutation_strategy not in allowed_mutations:
            raise (InvalidInput(f"{mutation_strategy} is an invalid mutation strategy"))

        self.G = graph
        self.mutation_strategy = mutation_strategy
        self.fitness_time = 0
        self.chromosomes = defaultdict(int)

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

        mutation_rows, mutation_cols = self.get_mutation_rows_cols(
            adjusted_n_mutations, population
        )

        mutation_strategy = self.mutation_strategy
        if "mutation_strategy" in kwargs:
            mutation_strategy = kwargs["mutation_strategy"]

        if mutation_strategy == "2-opt":

            return self.two_opt_mutation(population, mutation_rows)

        # elif mutation_strategy == '3-opt':
        #
        #     return self.random_inversion_mutation(population, mutation_rows, 3)

        elif mutation_strategy == "random_swap":

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
                population, mutation_rows, np.random.randint(2, population.shape[1] / 2)
            )

        elif mutation_strategy == "select_any_mutation":

            selected_strategy = np.random.choice(list(mutation_options), 1)[0]

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
