import logging
import pickle

import pytest

import numpy as np

from geneal.applications.fitness_functions.continuous import (
    fitness_functions_continuous,
)
from geneal.genetic_algorithms import BinaryGenAlgSolver, ContinuousGenAlgSolver
from geneal.utils.exceptions import InvalidInput, NoFitnessFunction


def quiet(**kwargs):
    return dict(verbose=False, show_stats=False, plot_results=False, **kwargs)


def run_ask_tell(solver, fitness_function):
    while not solver.done:
        population = solver.ask()
        solver.tell([fitness_function(individual) for individual in population])


class TestAskTell:
    @pytest.mark.parametrize(
        "solver_class, kwargs",
        [
            pytest.param(
                ContinuousGenAlgSolver,
                dict(n_genes=4, variables_limits=(-10, 10)),
                id="continuous",
            ),
            pytest.param(
                ContinuousGenAlgSolver,
                dict(
                    n_genes=4,
                    variables_limits=(0, 9),
                    variables_type=(int, "categorical", float, "categorical"),
                    fitness_tolerance=(1e-6, 5),
                ),
                id="continuous-mixed_types-tolerance",
            ),
            pytest.param(
                BinaryGenAlgSolver,
                dict(n_genes=12, selection_strategy="tournament"),
                id="binary",
            ),
        ],
    )
    def test_matches_solve(self, solver_class, kwargs):

        fitness_function = lambda x: -np.sum((x - 3) ** 2)

        expected = solver_class(
            fitness_function=fitness_function,
            pop_size=12,
            max_gen=30,
            random_state=7,
            **quiet(**kwargs),
        )
        expected.solve()

        solver = solver_class(pop_size=12, max_gen=30, random_state=7, **quiet(**kwargs))
        run_ask_tell(solver, fitness_function)

        assert solver.generations_ == expected.generations_
        assert solver.best_fitness_ == expected.best_fitness_
        assert np.array_equal(solver.best_individual_, expected.best_individual_)
        assert np.array_equal(solver.population_, expected.population_)
        assert np.array_equal(solver.mean_fitness_, expected.mean_fitness_)

    def test_lockstep_matches_individual_runs(self):

        targets = [np.array([1.0, 2.0]), np.array([-3.0, 4.0]), np.array([5.0, -5.0])]

        solvers = [
            ContinuousGenAlgSolver(n_genes=2, pop_size=10, max_gen=20, random_state=i, **quiet())
            for i in range(len(targets))
        ]

        def batched_fitness(population, target_ids):
            return -np.sum((population - np.array(targets)[target_ids]) ** 2, axis=1)

        n_batches = 0
        while not all(solver.done for solver in solvers):
            active = [i for i, solver in enumerate(solvers) if not solver.done]
            candidates = [solvers[i].ask() for i in active]
            target_ids = np.repeat(active, [len(c) for c in candidates])

            fitness = batched_fitness(np.vstack(candidates), target_ids)
            n_batches += 1

            splits = np.cumsum([len(c) for c in candidates])[:-1]
            for i, solver_fitness in zip(active, np.split(fitness, splits)):
                solvers[i].tell(solver_fitness)

        assert n_batches == 21

        for i, target in enumerate(targets):
            expected = ContinuousGenAlgSolver(
                n_genes=2,
                fitness_function=lambda x, t=target: -np.sum((x - t) ** 2),
                pop_size=10,
                max_gen=20,
                random_state=i,
                **quiet(),
            )
            expected.solve()

            assert np.array_equal(solvers[i].best_individual_, expected.best_individual_)
            assert solvers[i].best_fitness_ == expected.best_fitness_

    def test_ask_returns_individuals_to_evaluate(self):

        solver = ContinuousGenAlgSolver(n_genes=3, pop_size=8, max_gen=2, random_state=1, **quiet())

        with pytest.raises(RuntimeError):
            solver.tell(np.zeros(8))

        population = solver.ask()
        assert population.shape == (8, 3)
        assert np.array_equal(solver.ask(), population)

        with pytest.raises(InvalidInput):
            solver.tell(np.zeros(7))

        solver.tell(-np.abs(population).sum(axis=1))
        assert solver.generations_ == 0
        assert not solver.done

        offspring = solver.ask()
        assert offspring.shape == (7, 3)

        solver.tell(-np.abs(offspring).sum(axis=1))
        assert solver.generations_ == 1

        solver.tell(-np.abs(solver.ask()).sum(axis=1))
        assert solver.generations_ == 2
        assert solver.done

        with pytest.raises(RuntimeError):
            solver.ask()

    def test_no_fitness_function_needed(self):

        class BatchedSolver(ContinuousGenAlgSolver):
            def calculate_fitness(self, population):
                return -np.sum(population ** 2, axis=1)

        solver = BatchedSolver(n_genes=3, pop_size=10, max_gen=5, random_state=3, **quiet())
        solver.solve()

        assert solver.generations_ == 5

        with pytest.raises(NoFitnessFunction):
            ContinuousGenAlgSolver(n_genes=3, pop_size=10, max_gen=5, **quiet()).solve()


class TestInitialPopulation:
    def test_seeds_first_individuals(self):

        seeds = np.array([[1, 2, 3], [4, 5, 6]])

        solver = ContinuousGenAlgSolver(
            n_genes=3,
            pop_size=6,
            variables_limits=(0, 9),
            variables_type="categorical",
            initial_population=seeds,
            random_state=0,
            **quiet(),
        )

        population = solver.ask()

        assert np.array_equal(population[:2], seeds)
        assert population.shape == (6, 3)

    def test_optimal_seed_is_kept(self):

        target = np.array([1.5, -2.5, 3.5])

        solver = ContinuousGenAlgSolver(
            n_genes=3,
            fitness_function=lambda x: -np.sum((x - target) ** 2),
            pop_size=10,
            max_gen=10,
            initial_population=target,
            random_state=0,
            **quiet(),
        )
        solver.solve()

        assert np.array_equal(solver.best_individual_, target)
        assert solver.best_fitness_ == 0

    @pytest.mark.parametrize(
        "initial_population",
        [
            pytest.param(np.zeros((2, 4)), id="wrong_n_genes"),
            pytest.param(np.zeros((11, 3)), id="larger_than_pop_size"),
            pytest.param(np.zeros((0, 3)), id="empty"),
        ],
    )
    def test_invalid_shape(self, initial_population):

        with pytest.raises(InvalidInput):
            ContinuousGenAlgSolver(n_genes=3, pop_size=10, initial_population=initial_population)


class TestRandomState:
    def test_global_random_state_is_untouched(self):

        np.random.seed(0)
        state = np.random.get_state()

        solver = ContinuousGenAlgSolver(
            n_genes=3,
            fitness_function=lambda x: -np.sum(x ** 2),
            pop_size=10,
            max_gen=5,
            random_state=42,
            **quiet(),
        )
        solver.solve()

        new_state = np.random.get_state()
        assert new_state[2] == state[2]
        assert np.array_equal(new_state[1], state[1])

    def test_random_state_instance(self):

        results = []
        for _ in range(2):
            solver = ContinuousGenAlgSolver(
                n_genes=3,
                fitness_function=fitness_functions_continuous(3),
                pop_size=10,
                max_gen=5,
                random_state=np.random.RandomState(5),
                **quiet(),
            )
            solver.solve()
            results.append(solver.best_individual_)

        assert np.array_equal(*results)

    @pytest.mark.parametrize("random_state", [None, 3], ids=["None", "int"])
    def test_solver_can_be_pickled(self, random_state):

        solver = ContinuousGenAlgSolver(
            n_genes=3, pop_size=10, max_gen=3, random_state=random_state, **quiet()
        )
        solver.tell(-np.sum(solver.ask() ** 2, axis=1))

        clone = pickle.loads(pickle.dumps(solver))

        assert np.array_equal(clone.population_, solver.population_)
        assert np.array_equal(clone.ask(), solver.ask())

    def test_invalid_random_state(self):

        with pytest.raises(InvalidInput):
            ContinuousGenAlgSolver(n_genes=3, random_state="seed")


class TestLogging:
    def test_root_logger_handlers_are_untouched(self):

        root_logger = logging.getLogger()
        handler = logging.NullHandler()
        root_logger.addHandler(handler)

        try:
            ContinuousGenAlgSolver(n_genes=3, fitness_function=lambda x: x.sum())

            assert handler in root_logger.handlers
        finally:
            root_logger.removeHandler(handler)
