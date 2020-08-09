import pytest

import numpy as np

from geneal.applications.fitness_functions.binary import fitness_functions_binary
from geneal.genetic_algorithms import BinaryGenAlgSolver
from tests.mock_fixtures.mock_fixtures import mock_matplotlib, mock_logging


class TestBinaryGenAlgSolver:
    def test_initialize_population(self):

        binary_solver = BinaryGenAlgSolver(
            fitness_function=lambda x: x.sum(), n_genes=4, pop_size=5, random_state=42
        )

        population = binary_solver.initialize_population()

        expected_population = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
            ]
        )

        print(expected_population)

        assert np.equal(population, expected_population).all()

    @pytest.mark.parametrize(
        "crossover_pt, expected_result",
        [
            pytest.param(
                np.array([0]),
                np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1]),
                id="crossover_point=0",
            ),
            pytest.param(
                np.array([4]),
                np.array([1, 0, 0, 1, 0, 1, 1, 0, 1, 1]),
                id="crossover_point=4",
            ),
            pytest.param(
                np.array([10]),
                np.array([1, 0, 0, 1, 1, 1, 0, 1, 1, 0]),
                id="crossover_point=9",
            ),
        ],
    )
    def test_create_offspring(self, crossover_pt, expected_result):

        first_parent = np.array([1, 0, 0, 1, 1, 1, 0, 1, 1, 0])
        sec_parent = np.array([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])

        offspring = BinaryGenAlgSolver.create_offspring(
            first_parent, sec_parent, crossover_pt, None
        )

        print(offspring)

        assert np.equal(offspring, expected_result).all()

    def test_mutate_population(self):

        binary_solver = BinaryGenAlgSolver(
            fitness_function=lambda x: x.sum(), n_genes=4, pop_size=5, random_state=42
        )

        population = binary_solver.initialize_population()

        expected_mutated_population = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )

        mutated_population = binary_solver.mutate_population(population, 5)

        print(mutated_population)

        assert np.equal(mutated_population, expected_mutated_population).all()

    @pytest.mark.parametrize(
        "fitness_function, n_genes, expected_best_fitness, expected_best_individual",
        [
            pytest.param(
                1,
                50,
                47.0,
                np.array(
                    [
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                    ]
                ),
                id="binary_fitness_function=1",
            ),
            pytest.param(
                2,
                48,
                -4.0,
                np.array(
                    [
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                id="binary_fitness_function=2",
            ),
        ],
    )
    def test_solve(
        self,
        mock_matplotlib,
        mock_logging,
        fitness_function,
        n_genes,
        expected_best_fitness,
        expected_best_individual,
    ):

        solver = BinaryGenAlgSolver(
            n_genes,
            fitness_functions_binary(fitness_function),
            pop_size=10,
            max_gen=50,
            mutation_rate=0.05,
            selection_rate=0.5,
            random_state=42,
        )

        solver.solve()

        print(solver.best_individual_)

        assert solver.best_fitness_ == expected_best_fitness
        assert np.equal(solver.best_individual_, expected_best_individual).all()
