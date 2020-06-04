from functools import reduce

import numpy as np
import networkx as nx

from geneal.genetic_algorithms import ContinuousGenAlgSolver


class TSP(ContinuousGenAlgSolver):
    def __init__(self, graph, *args, **kwargs):

        if "n_crossover_points" in kwargs:
            if kwargs["n_crossover_points"] != 2:
                print("Defaulting 'n_crossover_points' to 2")
            kwargs.pop("n_crossover_points")

        ContinuousGenAlgSolver.__init__(self, n_crossover_points=2, *args, **kwargs)

        self.G = graph

    def fitness_function(self, individual):
        res = reduce(
            lambda total_length, city_pair: total_length
            + self.G.edges[(city_pair[0], city_pair[1])]["weight"],
            zip(individual, individual[1:]),
            0,
        )

        res += self.G.edges[(241, individual[0])]["weight"]
        res += self.G.edges[(individual[-1], 241)]["weight"]

        return -round(res, 2)

    def initialize_population(self):

        population = np.repeat(
            np.arange(1, self.n_genes + 1)[np.newaxis, :], self.pop_size, axis=0
        )

        return np.array(list(map(lambda x: np.random.permutation(x), population)))

    def create_offspring(self, first_parent, sec_parent, crossover_pt, _):

        reordered_sec_parent = np.roll(sec_parent, -crossover_pt[1])

        new_arr = first_parent[crossover_pt[0] : crossover_pt[1]]

        new_arr = np.append(new_arr, reordered_sec_parent)

        _, idx = np.unique(new_arr, return_index=True)

        res = np.roll(new_arr[np.sort(idx)], crossover_pt[0])

        if res.shape[0] != 30:
            a = 1

        return res

    def mutate_population_inverse(self, population, n_mutations):

        adjusted_n_mutations = np.ceil(n_mutations / self.n_genes).astype(int)

        mutation_rows, mutation_cols = self.get_mut_rows_cols(
            adjusted_n_mutations, population
        )

        population[mutation_rows, :] = np.array(
            list(
                map(
                    lambda args: self.mutation_helper(*args),
                    zip(population[mutation_rows, :], mutation_cols),
                )
            )
        )

        return population

    def mutate_population(self, population, n_mutations):

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
        route[mutation_cols[0] : mutation_cols[1]] = np.flip(
            route[mutation_cols[0] : mutation_cols[1]]
        )

        return route

    def mutation_helper(self, row, mutation_cols):
        row[mutation_cols[0] : mutation_cols[1]] = np.flip(
            row[mutation_cols[0] : mutation_cols[1]]
        )

        chosen_gene_index = np.random.choice(np.arange(row.shape[0]), 1)[0]
        chosen_gene = row[chosen_gene_index]

        closest_neighbour = list(nx.neighbors(G, chosen_gene))[0]

        chosen_neighbour = np.random.choice(
            list(nx.neighbors(self.G, closest_neighbour))[:5], 1
        )[0]

        chosen_neighbour_index = np.argwhere(row == chosen_neighbour)[0, 0]

        (row[chosen_gene_index], row[chosen_neighbour_index]) = (
            row[chosen_neighbour_index],
            row[chosen_gene_index],
        )

        return row

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
