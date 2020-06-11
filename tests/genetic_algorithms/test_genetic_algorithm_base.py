import pytest

import numpy as np

from geneal.genetic_algorithms.genetic_algorithm_base import GenAlgSolver
from geneal.utils.exceptions import NoFitnessFunction


class TestGenAlgSolver:
    def test_mutate_population(self):

        gen_alg = GenAlgSolver(
            fitness_function=lambda x: x.sum(), n_genes=10, pop_size=10, random_state=42
        )

        mutation_rows, mutation_cols = gen_alg.mutate_population(None, 10)

        assert np.equal(mutation_rows, np.array([[4, 9, 7, 6, 2, 2, 1, 8, 6, 7]])).all()
        assert np.equal(mutation_cols, np.array([[0, 9, 8, 2, 1, 1, 3, 5, 4, 2]])).all()

    @pytest.mark.parametrize(
        "n_crossover_points, expected_result",
        [
            pytest.param(1, np.array([5]), id="n_crossover_points=1"),
            pytest.param(2, np.array([0, 5]), id="n_crossover_points=2"),
            pytest.param(3, np.array([[0, 5, 9]]), id="n_crossover_points=3"),
        ],
    )
    def test_get_crossover_points(self, n_crossover_points, expected_result):

        gen_alg = GenAlgSolver(
            fitness_function=lambda x: x.sum(),
            n_genes=10,
            n_crossover_points=n_crossover_points,
            random_state=42,
        )

        crossover_points = gen_alg.get_crossover_points()

        assert np.equal(crossover_points, expected_result).all()

    def test_no_fitness_function_error(self):

        with pytest.raises(Exception) as excinfo:
            GenAlgSolver(n_genes=10, random_state=42)

        assert excinfo.type == NoFitnessFunction
        assert (
            str(excinfo.value)
            == "A fitness function must be defined or provided as an argument"
        )

    @pytest.mark.parametrize(
        "pop_size, selection_strategy, expected_ma, expected_pa",
        [
            pytest.param(
                10,
                "roulette_wheel",
                np.array([0, 4]),
                np.array([3, 0]),
                id="roulette_wheel-pop_size=10",
            ),
            pytest.param(
                11,
                "roulette_wheel",
                np.array([4, 3, 0]),
                np.array([0, 0, 0]),
                id="roulette_wheel-pop_size=11",
            ),
            pytest.param(
                10,
                "two_by_two",
                np.array([0, 2]),
                np.array([1, 3]),
                id="two_by_two-pop_size=10",
            ),
            pytest.param(
                11,
                "two_by_two",
                np.array([0, 2, 4]),
                np.array([1, 3, 5]),
                id="two_by_two-pop_size=11",
            ),
            pytest.param(
                10,
                "random",
                np.array([0, 4]),
                np.array([[4, 1]]),
                id="random-pop_size=10",
            ),
            pytest.param(
                11,
                "random",
                np.array([4, 4, 1]),
                np.array([[0, 0, 1]]),
                id="random-pop_size=11",
            ),
            pytest.param(
                10,
                "tournament",
                np.array([1, 1]),
                np.array([[1, 3]]),
                id="tournament-pop_size=10",
            ),
            pytest.param(
                11,
                "tournament",
                np.array([1, 1, 3]),
                np.array([[1, 3, 1]]),
                id="tournament-pop_size=11",
            ),
        ],
    )
    def test_make_selection(
        self, pop_size, selection_strategy, expected_ma, expected_pa
    ):

        np.random.seed(42)

        n_genes = 10

        gen_alg = GenAlgSolver(
            fitness_function=lambda x: x.sum(),
            pop_size=pop_size,
            selection_strategy=selection_strategy,
            n_genes=n_genes,
            random_state=42,
        )

        fitness = np.random.rand(pop_size, 1)

        ma, pa = gen_alg.select_parents(fitness)

        assert np.allclose(ma, expected_ma)
        assert np.allclose(pa, expected_pa)
