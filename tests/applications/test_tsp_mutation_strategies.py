import pytest

import numpy as np

from geneal.applications.tsp.travelling_salesman_problem import (
    TravellingSalesmanProblemSolver,
)
from tests.applications.fixtures.tsp_test_fixture import G


class TestMutationStrategies:

    def test_two_opt_mutation(self):

        pop_size = 8

        tsp_solver = TravellingSalesmanProblemSolver(
            G, pop_size=pop_size, random_state=42,
        )

        route = np.array([2, 1, 5, 7, 3, 4, 6, 8])

        expected_mutated_route = np.array([6, 8, 2, 1, 4, 3, 7, 5])

        mutated_route = tsp_solver.two_opt_mutation_helper(route)

        assert np.allclose(mutated_route, expected_mutated_route)

    def test_random_inversion_mutation(self):

        pop_size = 8

        tsp_solver = TravellingSalesmanProblemSolver(
            G, pop_size=pop_size, random_state=42,
        )

        route = np.array([2, 1, 5, 7, 3, 4, 6, 8])
        mutation_cols = tsp_solver.get_consecutive_mutation_cols(route, 1, 5)

        expected_mutated_route = np.array([2, 8, 6, 7, 3, 4, 5, 1])

        mutated_route = tsp_solver.random_inversion_mutation_helper(
            route, mutation_cols
        )

        assert np.allclose(mutated_route, expected_mutated_route)

    def test_random_gene_around_nearest_neighbour_mutation(self):

        pop_size = 8

        tsp_solver = TravellingSalesmanProblemSolver(
            G, pop_size=pop_size, random_state=42,
        )

        route = np.array([2, 1, 5, 7, 3, 4, 6, 8])

        expected_mutated_route = np.array([2, 1, 5, 7, 6, 4, 3, 8])

        mutated_route = tsp_solver.random_gene_around_nearest_neighbour_mutation_helper(
            route
        )

        assert np.allclose(mutated_route, expected_mutated_route)

    def test_random_swap_mutation(self):

        pop_size = 8

        tsp_solver = TravellingSalesmanProblemSolver(
            G, pop_size=pop_size, random_state=42,
        )

        route = np.array([[2, 1, 5, 7, 3, 4, 6, 8]])

        expected_mutated_route = np.array([2, 3, 5, 7, 1, 4, 6, 8])

        mutation_rows = np.array([0])
        mutation_cols = np.array([[1, 4]])

        mutated_route = tsp_solver.random_swap_mutation(
            route, mutation_rows, mutation_cols
        )

        assert np.allclose(mutated_route, expected_mutated_route)

    def test_worst_gene_random_mutation(self):

        pop_size = 8

        tsp_solver = TravellingSalesmanProblemSolver(
            G, pop_size=pop_size, random_state=42,
        )

        route = np.array([2, 1, 5, 7, 3, 4, 6, 8])

        expected_mutated_route = np.array([2, 1, 5, 7, 3, 4, 8, 6])

        mutated_route = tsp_solver.worst_gene_random_mutation_helper(route)

        assert np.allclose(mutated_route, expected_mutated_route)

    def test_worst_gene_nearest_neighbour_mutation(self):

        pop_size = 8

        tsp_solver = TravellingSalesmanProblemSolver(
            G, pop_size=pop_size, random_state=42,
        )

        route = np.array([2, 1, 5, 7, 3, 4, 6, 8])

        expected_mutated_route = np.array([2, 1, 8, 7, 3, 4, 6, 5])

        mutated_route = tsp_solver.worst_gene_nearest_neighbour_mutation_helper(route)

        assert np.allclose(mutated_route, expected_mutated_route)

    def test_random_gene_nearest_neighbour(self):

        pop_size = 8

        tsp_solver = TravellingSalesmanProblemSolver(
            G, pop_size=pop_size, random_state=42,
        )

        route = np.array([2, 1, 5, 7, 3, 4, 6, 8])

        expected_mutated_route = np.array([2, 6, 1, 5, 7, 3, 4, 8])

        mutated_route = tsp_solver.random_gene_nearest_neighbour_helper(route)

        assert np.allclose(mutated_route, expected_mutated_route)
