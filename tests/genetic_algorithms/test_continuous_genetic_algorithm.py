import pytest

import numpy as np

from geneal.applications.fitness_functions.continuous import (
    fitness_functions_continuous,
)
from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.utils.exceptions import InvalidInput
from tests.mock_fixtures.mock_fixtures import mock_matplotlib, mock_logging


class TestContinuousGenAlgSolver:

    @pytest.mark.parametrize(
        "variables_type",
        [
            pytest.param(
                (int, int, int),
                id="variables_type-wrong_dimensions",
            ),
            pytest.param(
                "variables_type",
                id="variables_type-invalid",
            )
        ],
    )
    def test_invalid_input(self, variables_type):
        with pytest.raises(Exception) as excinfo:
            continuous_solver = ContinuousGenAlgSolver(
                fitness_function=lambda x: x.sum(),
                n_genes=4,
                pop_size=5,
                variables_type=variables_type,
                random_state=42,
            )

        assert excinfo.type == InvalidInput

    @pytest.mark.parametrize(
        "variables_type,variable_limits",
        [
            pytest.param(
                None,
                [-10, 10],
                id="variables_type-None",
            ),
            pytest.param(
                [float, float, float],
                [-10, 10],
                id="variables_type-correct_tuple",
            ),
            pytest.param(
                None,
                None,
                id="variable_limits-None",
            ),
        ],
    )
    def test_valid_input(self, variables_type, variable_limits):

        n_genes = 3

        continuous_solver = ContinuousGenAlgSolver(
            fitness_function=lambda x: x.sum(),
            n_genes=n_genes,
            pop_size=5,
            variables_type=variables_type,
            variables_limits=variable_limits,
            random_state=42,
        )

        assert continuous_solver.variables_type == [float] * n_genes

        if variable_limits is not None:
            assert continuous_solver.variables_limits == [variable_limits] * n_genes
        else:
            min_max = np.iinfo(np.int64)
            assert continuous_solver.variables_limits == [(min_max.min, min_max.max)] * n_genes

    @pytest.mark.parametrize(
        "variables_type, expected_result",
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
                id="variables_type=int",
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
                id="variables_type=float",
            ),
        ],
    )
    def test_initialize_population(self, variables_type, expected_result):

        continuous_solver = ContinuousGenAlgSolver(
            fitness_function=lambda x: x.sum(),
            n_genes=4,
            pop_size=5,
            variables_type=variables_type,
            random_state=42,
        )

        population = continuous_solver.initialize_population()

        assert np.allclose(population, expected_result, rtol=1e-05)

    @pytest.mark.parametrize(
        "variables_limits, variables_type",
        [
            pytest.param(
                (-20, 20),
                int,
                id="variables_type=int | variables_limits=same for all genes",
            ),
            pytest.param(
                [(0, 10), (-10, 5), (-100, 100), (5, 20)],
                float,
                id="variables_type=float | variables_limits=different for all genes",
            ),
        ],
    )
    def test_initialize_population_variable_limits(
        self, variables_limits, variables_type
    ):
        continuous_solver = ContinuousGenAlgSolver(
            fitness_function=lambda x: x.sum(),
            n_genes=4,
            pop_size=5,
            variables_type=variables_type,
            variables_limits=variables_limits,
            random_state=42,
        )

        population = continuous_solver.initialize_population()

        for i in range(population.shape[1]):
            assert population[:, i].min() > continuous_solver.variables_limits[i][0]
            assert population[:, i].max() < continuous_solver.variables_limits[i][1]

    @pytest.mark.parametrize(
        "variables_type,crossover_pt,expected_first_offspring,expected_second_offspring",
        [
            pytest.param(
                float,
                np.array([0]),
                np.array([1.37454012, 2, -4, 0]),
                np.array([1.62545988, -3.0, 5.0, 0.0]),
                id="float-crossover_point=0",
            ),
            pytest.param(
                float,
                np.array([2]),
                np.array([1, -3, 1.62913893, 0]),
                np.array([[2.0, 2.0, -0.62913893, 0.0]]),
                id="float-crossover_point=4",
            ),
            pytest.param(
                float,
                np.array([3]),
                np.array([1.0, -3.0, 5.0, 0.0]),
                np.array([2.0, 2.0, -4.0, 0.0]),
                id="float-crossover_point=9",
            ),
            pytest.param(
                int,
                np.array([0]),
                [1,  2, - 4,  0],
                [2, -3,  5,  0],
                id="int-crossover_point=0",
            ),
            pytest.param(
                int,
                np.array([2]),
                [1, - 3,  2,  0],
                [2,  2, - 1,  0],
                id="int-crossover_point=4",
            ),
        ],
    )
    def test_create_offspring_float(
        self, variables_type, crossover_pt, expected_first_offspring, expected_second_offspring
    ):

        continuous_solver = ContinuousGenAlgSolver(
            fitness_function=lambda x: x.sum(), n_genes=4, pop_size=5, random_state=42, variables_type=variables_type
        )

        first_parent = np.array([1, -3, 5, 0])
        sec_parent = np.array([2, 2, -4, 0])

        first_offspring = continuous_solver.create_offspring(
            first_parent, sec_parent, crossover_pt, "first"
        )

        second_offspring = continuous_solver.create_offspring(
            sec_parent, first_parent, crossover_pt, "second"
        )

        print(first_offspring)
        print(second_offspring)

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
                [9.01428613, -8.83832776, 9.39819704, -3.91515514],
                [4.63987884, 7.32352292, -8.04655772, 0.49512863],
                [1.97316968, 2.02230023, -5.75321779, 8.18640804],
                [-6.87962719, 9.31264066, -6.36350066, -4.1754172],
            ]
        )

        mutated_population = continuous_solver.mutate_population(population, 5)

        print(mutated_population)

        assert np.allclose(mutated_population, expected_mutated_population, rtol=1e-5)

    @pytest.mark.parametrize(
        'fitness_tolerance',
        [
            pytest.param(
                None,
                id='no_tolerance'
            ),
            pytest.param(
                (10, 2),
                id='with_tolerance'
            ),
        ]
    )
    @pytest.mark.parametrize(
        "fitness_function, n_genes, expected_best_fitness, expected_best_individual",
        [
            pytest.param(
                1, 1, -1.0817, np.array([0.40]), id="continuous_fitness_function=1",
            ),
            pytest.param(
                2, 1, -0.653, np.array([-0.87]), id="continuous_fitness_function=2",
            ),
            pytest.param(
                3,
                2,
                -4.09,
                np.array([2.011, -1.361]),
                id="continuous_fitness_function=3",
            ),
            pytest.param(
                4,
                3,
                17.0614,
                np.array([4.632, -4.175, -6.025]),
                id="continuous_fitness_function=4",
            ),
            pytest.param(
                5, 1, 248.3267, np.array([16.172]), id="continuous_fitness_function=5",
            ),
            pytest.param(
                6,
                2,
                18.2684,
                np.array([[9.014, 9.398]]),
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
        fitness_tolerance
    ):

        solver = ContinuousGenAlgSolver(
            n_genes,
            fitness_functions_continuous(fitness_function),
            pop_size=10,
            max_gen=2,
            mutation_rate=0.05,
            selection_rate=0.5,
            random_state=42,
            fitness_tolerance=fitness_tolerance
        )

        solver.solve()

        assert np.allclose(solver.best_fitness_, expected_best_fitness, rtol=1)
        assert np.allclose(solver.best_individual_, expected_best_individual, rtol=1)
