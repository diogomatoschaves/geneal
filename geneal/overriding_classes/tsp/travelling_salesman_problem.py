from functools import reduce

import numpy as np


def dist(xy1, xy2):
    return np.sqrt((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2)


cities = {
    1: {"coords": [392.8, 356.4], "name": "Oklahoma City"},
    2: {"coords": [559.6, 404.8], "name": "Montgomery"},
    3: {"coords": [451.6, 186.0], "name": "Saint Paul"},
    4: {"coords": [698.8, 239.6], "name": "Trenton"},
    5: {"coords": [204.0, 243.2], "name": "Salt Lake City"},
    6: {"coords": [590.8, 263.2], "name": "Columbus"},
    7: {"coords": [389.2, 448.4], "name": "Austin"},
    8: {"coords": [179.6, 371.2], "name": "Phoenix"},
    9: {"coords": [719.6, 205.2], "name": "Hartford"},
    10: {"coords": [489.6, 442.0], "name": "Baton Rouge"},
    11: {"coords": [80.0, 139.2], "name": "Salem"},
    12: {"coords": [469.2, 367.2], "name": "Little Rock"},
    13: {"coords": [673.2, 293.6], "name": "Richmond"},
    14: {"coords": [501.6, 409.6], "name": "Jackson"},
    15: {"coords": [447.6, 246.0], "name": "Des Moines"},
    16: {"coords": [563.6, 216.4], "name": "Lansing"},
    17: {"coords": [293.6, 274.0], "name": "Denver"},
    18: {"coords": [159.6, 182.8], "name": "Boise"},
    19: {"coords": [662.0, 328.8], "name": "Raleigh"},
    20: {"coords": [585.6, 376.8], "name": "Atlanta"},
    21: {"coords": [500.8, 217.6], "name": "Madison"},
    22: {"coords": [548.0, 272.8], "name": "Indianapolis"},
    23: {"coords": [546.4, 336.8], "name": "Nashville"},
    24: {"coords": [632.4, 364.8], "name": "Columbia"},
    25: {"coords": [735.2, 201.2], "name": "Providence"},
    26: {"coords": [738.4, 190.8], "name": "Boston"},
    27: {"coords": [594.8, 434.8], "name": "Tallahassee"},
    28: {"coords": [68.4, 254.0], "name": "Sacramento"},
    29: {"coords": [702.0, 193.6], "name": "Albany"},
    30: {"coords": [670.8, 244.0], "name": "Harrisburg"},
}


class TSP:

    @staticmethod
    def fitness_function(individual):
        res = reduce(
            lambda total_length, city_pair: total_length
            + dist(cities[city_pair[0]]["coords"], cities[city_pair[1]]["coords"]),
            zip(individual, individual[1:]),
            0,
        )

        res += dist(cities[individual[0]]["coords"], cities[individual[-1]]["coords"])

        return -round(res, 2)

    def initialize_population(self):

        population = np.repeat(
            np.arange(1, self.n_genes + 1)[np.newaxis, :], self.pop_size, axis=0
        )

        return np.array(list(map(lambda x: np.random.permutation(x), population)))

    def create_offspring(self, first_parent, sec_parent, crossover_pt, _):

        reordered_sec_parent = np.roll(sec_parent, -crossover_pt[1])

        new_arr = first_parent[crossover_pt[0]: crossover_pt[1]]

        new_arr = np.append(new_arr, reordered_sec_parent)

        _, idx = np.unique(new_arr, return_index=True)

        res = np.roll(new_arr[np.sort(idx)], crossover_pt[0])

        if res.shape[0] != 10:
            a = 1

        return res

    # def mutate_population(self, population, n_mutations):
    #
    #     adjusted_n_mutations = np.ceil(n_mutations / self.n_genes).astype(int)
    #
    #     mutation_rows = np.random.choice(
    #         np.arange(1, population.shape[0]),
    #         adjusted_n_mutations,
    #         replace=False
    #     )
    #
    #     mutation_cols = (
    #         np.ceil(np.random.rand(adjusted_n_mutations, 2) * self.n_genes) - 1
    #     ).astype(int)
    #
    #     (
    #         population[mutation_rows, mutation_cols[:, 0]],
    #         population[mutation_rows, mutation_cols[:, 1]],
    #     ) = (
    #         population[mutation_rows, mutation_cols[:, 1]],
    #         population[mutation_rows, mutation_cols[:, 0]],
    #     )
    #
    #     return population

    def mutate_population(self, population, n_mutations):

        adjusted_n_mutations = np.ceil(n_mutations / self.n_genes).astype(int)

        mutation_rows = np.random.choice(
            np.arange(1, population.shape[0]),
            adjusted_n_mutations,
            replace=False
        )

        mutation_cols = (
            np.ceil(np.random.rand(adjusted_n_mutations, 2) * self.n_genes) - 1
        ).astype(int)

        (
            population[mutation_rows, mutation_cols[:, 0]],
            population[mutation_rows, mutation_cols[:, 1]],
        ) = (
            population[mutation_rows, mutation_cols[:, 1]],
            population[mutation_rows, mutation_cols[:, 0]],
        )

        return population
