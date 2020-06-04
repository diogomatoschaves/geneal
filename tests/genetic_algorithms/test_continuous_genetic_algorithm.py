import pytest

import numpy as np

from geneal.applications.fitness_functions.continuous import (
    fitness_functions_continuous,
)
from geneal.genetic_algorithms import ContinuousGenAlgSolver
from tests.mock_fixtures.mock_fixtures import mock_matplotlib, mock_logging


class TestContinuousGenAlgSolver:
    @pytest.mark.parametrize(
        "problem_type, expected_result",
        [
            pytest.param(
                int,
                np.array(
                    [
                        [-4.0, 10.0, 10.0, -9.0],
                        [9.0, -4.0, -7.0, 1.0],
                        [4.0, 8.0, -3.0, -5.0],
                        [0.0, 0.0, -8.0, -9.0],
                        [-3.0, 0.0, 10.0, 10.0],
                    ]
                ),
                id="problem_type=int",
            ),
            pytest.param(
                float,
                np.array(
                    [
                        [-2.50919762, -6.88010959, -9.58831011, -6.3319098],
                        [9.01428613, -8.83832776, 9.39819704, -3.91515514],
                        [4.63987884, 7.32352292, 6.64885282, 0.49512863],
                        [1.97316968, 2.02230023, -5.75321779, -1.36109963],
                        [-6.87962719, 4.16145156, -6.36350066, -4.1754172],
                    ]
                ),
                id="problem_type=float",
            ),
        ],
    )
    def test_initialize_population(self, problem_type, expected_result):

        continuous_solver = ContinuousGenAlgSolver(
            fitness_function=lambda x: x.sum(),
            n_genes=4,
            pop_size=5,
            problem_type=problem_type,
            random_state=42,
        )

        population = continuous_solver.initialize_population()

        assert np.allclose(population, expected_result, rtol=1e-05)

    @pytest.mark.parametrize(
        "variables_limits, problem_type",
        [
            pytest.param(
                (-20, 20),
                int,
                id="problem_type=int | variables_limits=same for all genes",
            ),
            pytest.param(
                [(0, 10), (-10, 5), (-100, 100), (5, 20)],
                float,
                id="problem_type=float | variables_limits=different for all genes",
            ),
        ],
    )
    def test_initialize_population_variable_limits(
        self, variables_limits, problem_type
    ):

        continuous_solver = ContinuousGenAlgSolver(
            fitness_function=lambda x: x.sum(),
            n_genes=4,
            pop_size=5,
            problem_type=problem_type,
            variables_limits=variables_limits,
            random_state=42,
        )

        population = continuous_solver.initialize_population()

        for i in range(population.shape[1]):
            assert population[:, i].min() > continuous_solver.variables_limits[i][0]
            assert population[:, i].max() < continuous_solver.variables_limits[i][1]

    @pytest.mark.parametrize(
        "crossover_pt, expected_first_offspring, expected_second_offspring",
        [
            pytest.param(
                np.array([0]),
                np.array([1.37454012, 2, -4, 0]),
                np.array([2.95071431, -3.0, 5.0, 0.0]),
                id="crossover_point=0",
            ),
            pytest.param(
                np.array([2]),
                np.array([1, -3, 1.62913893, 0]),
                np.array([[2.0, 2.0, -12.55642876, 0.0]]),
                id="crossover_point=4",
            ),
            pytest.param(
                np.array([3]),
                np.array([1.0, -3.0, 5.0, 0.0]),
                np.array([2.0, 2.0, -4.0, 0.0]),
                id="crossover_point=9",
            ),
        ],
    )
    def test_create_offspring(
        self, crossover_pt, expected_first_offspring, expected_second_offspring
    ):

        continuous_solver = ContinuousGenAlgSolver(
            fitness_function=lambda x: x.sum(), n_genes=4, pop_size=5, random_state=42
        )

        first_parent = np.array([1, -3, 5, 0])
        sec_parent = np.array([2, 2, -4, 0])

        first_offspring = continuous_solver.create_offspring(
            first_parent, sec_parent, crossover_pt, "first"
        )

        second_offspring = continuous_solver.create_offspring(
            sec_parent, first_parent, crossover_pt, "second"
        )

        assert np.allclose(first_offspring, expected_first_offspring, rtol=1e-5)
        assert np.allclose(second_offspring, expected_second_offspring, rtol=1e-5)

    def test_mutate_population(self):

        continuous_solver = ContinuousGenAlgSolver(
            fitness_function=lambda x: x.sum(), n_genes=4, pop_size=5, random_state=42
        )

        population = continuous_solver.initialize_population()

        expected_mutated_population = np.array(
            [
                [-2.50919762, -6.88010959, -9.58831011, -6.3319098],
                [-6.58951753, -8.83832776, 9.39819704, -3.91515514],
                [-8.69896814, 7.32352292, -9.31222958, 0.49512863],
                [1.97316968, 2.02230023, -5.75321779, 0.93420559],
                [-6.87962719, 4.16145156, -6.36350066, -4.1754172],
            ]
        )

        mutated_population = continuous_solver.mutate_population(population, 5)

        assert np.allclose(mutated_population, expected_mutated_population, rtol=1e-5)

    @pytest.mark.parametrize(
        "fitness_function, n_genes, expected_best_fitness, expected_best_individual",
        [
            pytest.param(
                1, 1,  -1.0817, np.array([0.0854]), id="continuous_fitness_function=1",
            ),
            pytest.param(
                2,
                1,
                -3.078e-05,
                np.array([-0.0569]),
                id="continuous_fitness_function=2",
            ),
            pytest.param(
                3,
                2,
                -0.0046863,
                np.array([-0.009534,  0.0677]),
                id="continuous_fitness_function=3",
            ),
            pytest.param(
                4,
                3,
                15.407,
                np.array([-4.00808176,  4.60078633, -3.29044853]),
                id="continuous_fitness_function=4",
            ),
            pytest.param(
                5,
                1,
                227.423,
                np.array([16.348]),
                id="continuous_fitness_function=5",
            ),
            pytest.param(
                6,
                2,
                18.508,
                np.array([9.01428, 8.67935]),
                id="continuous_fitness_function=6",
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

        solver = ContinuousGenAlgSolver(
            n_genes,
            fitness_functions_continuous(fitness_function),
            pop_size=10,
            max_gen=50,
            mutation_rate=0.05,
            selection_rate=0.5,
            random_state=42,
        )

        solver.solve()

        print(solver.best_fitness_)

        assert np.allclose(solver.best_fitness_, expected_best_fitness, rtol=1e-2)
        assert np.allclose(solver.best_individual_, expected_best_individual, rtol=1e-2)
