# geneal

![build_badge](https://github.com/diogomatoschaves/geneal/workflows/build/badge.svg)
[![codecov](https://codecov.io/gh/diogomatoschaves/geneal/branch/master/graph/badge.svg)](https://codecov.io/gh/diogomatoschaves/geneal)
[![PyPI version](https://badge.fury.io/py/geneal.svg)](https://badge.fury.io/py/geneal)

`geneal` is a python library implementing genetic algorithms (GA). It has functionality for both binary and continuous GA,
as well as specific use case applications such as a solver for the
[Travelling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem). 

## Installation

```
$ pip install geneal
```

## Usage

`geneal` provides 2 main GA solver classes, `BinaryGenAlgSolver` and `ContinuousGenAlgSolver` 
for binary and continuous problem formulations respectively. The library is set up in such a way that all problems
have to be a maximization, and hence, the fitness functions need to be adjusted accordingly.

### Binary GA

The binary GA class can be customized by the input variables provided on its initialization. 
The minimum required arguments to provide at the initialization are the number of genes on each individual's
chromosome, and the fitness function to be maximized. 

An example is shown below.

```python

from geneal.genetic_algorithms import BinaryGenAlgSolver
from geneal.applications.fitness_functions.binary import fitness_functions_binary

solver = BinaryGenAlgSolver(
    n_genes=3, # number of variables defining the problem
    fitness_function=fitness_functions_binary(1), # fitness function to be maximized
)

solver.solve()

```

The above call will perform the optimization with the default parameters, which will most likely have to be adjusted for
each individual problem at hand. The algorithm parameters can be customized as shown below.

```python
from geneal.genetic_algorithms import BinaryGenAlgSolver
from geneal.applications.fitness_functions.binary import fitness_functions_binary

solver = BinaryGenAlgSolver(
    n_genes=3,
    fitness_function=fitness_functions_binary(1), 
    n_bits=1, # number of bits describing each gene (variable)
    pop_size=10, # population size (number of individuals)
    max_gen=500, # maximum number of generations
    mutation_rate=0.05, # mutation rate to apply to the population
    selection_rate=0.5, # percentage of the population to select for mating
)

solver.solve()

```

In the above call, we are customizing the GA run with specific parameters for our use case.

### Continuous GA

Following a similar pattern as the binary GA solver, the parameters of the continuous GA can be adjusted at initialization.
As a bare minimum, the number of variables and the fitness function to maximize must be provided, as shown below.

```python

from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.applications.fitness_functions.continuous import fitness_functions_continuous

solver = ContinuousGenAlgSolver(
    n_genes=4, # number of variables defining the problem
    fitness_function=fitness_functions_continuous(3), # fitness function to be maximized
)

solver.solve()

```

In order to customize the continuous GA solver further, more arguments can be passed at initialization.

```python

from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.applications.fitness_functions.continuous import fitness_functions_continuous

solver = ContinuousGenAlgSolver(
    n_genes=4, # number of variables defining the problem
    fitness_function=fitness_functions_continuous(3), # fitness function to be maximized
    pop_size=10, # population size (number of individuals)
    max_gen=200, # maximum number of generations
    mutation_rate=0.1, # mutation rate to apply to the population
    selection_rate=0.6, # percentage of the population to select for mating
)

solver.solve()

```

A notable difference to the binary GA solver is the fact that we can customize the input space of the problem
by defining if the problem is of type `int` or `float`, and defining an overall minimum or maximum value for all
variables (or individually for each variable).

```python

from geneal.genetic_algorithms import ContinuousGenAlgSolver
from geneal.applications.fitness_functions.continuous import fitness_functions_continuous

solver = ContinuousGenAlgSolver(
    n_genes=4, 
    fitness_function=fitness_functions_continuous(3),
    pop_size=10,
    max_gen=200,
    mutation_rate=0.1,
    selection_rate=0.6,
    problem_type=float, # Defines the possible values as float numbers
    variables_limits=(-10, 10) # Defines the limits of all variables between -10 and 10. 
                               # Alternatively one can pass an array of tuples defining the limits
                               # for each variable: [(-10, 10), (0, 5), (0, 5), (-20, 20)]
)

solver.solve()

```

## Specific Applications

In the real world, there's usually the need to adapt a genetic algorithm implementation to each individual problem.
Thus, `geneal` offers the user a level of customization that aims to be both versatile and relatively simple. For that,
one just has to create a class which inherits from the `BinaryGenAlgSolver` or `ContinuousGenAlgSolver`
base classes, and on which some overriding methods are defined. This allows the user to control the main steps of a GA:

- fitness function
- population initialization
- mating between individuals (creation of offsprings)
- mutation of the population
 
A boilerplate template of such a class is shown below:

```python
from geneal.genetic_algorithms import ContinuousGenAlgSolver, BinaryGenAlgSolver


class TemplateChildClass(ContinuousGenAlgSolver, BinaryGenAlgSolver):
    def __init__(self, *args, **kwargs):
        BinaryGenAlgSolver.__init__(self, *args, **kwargs)
        ContinuousGenAlgSolver.__init__(self, *args, **kwargs)

    def fitness_function(self, chromosome):
        """
        Implements the logic that calculates the fitness
        measure of an individual.

        :param chromosome: chromosome of genes representing an individual
        :return: the fitness of the individual
        """
        pass

    def initialize_population(self, pop_size, n_genes):
        """
        Initializes the population of the problem

        :param pop_size: number of individuals in the population
        :param n_genes: number of genes representing the problem. In case of the binary
        solver, it represents the number of genes times the number of bits per gene
        :return: a numpy array with a randomized initialized population
        """
        pass

    def create_offspring(
        self, first_parent, sec_parent, crossover_pt, offspring_number
    ):
        """
        Creates an offspring from 2 parents. It uses the crossover point(s)
        to determine how to perform the crossover

        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: point(s) at which to perform the crossover
        :param offspring_number: whether it's the first or second offspring from a pair of parents.
        Important if there's different logic to be applied to each case.
        :return: the resulting offspring.
        """
        pass

    def mutate_population(self, population, n_mutations):
        """
        Mutates the population according to a given user defined rule.

        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed. This number is 
        calculated according to mutation_rate, but can be adjusted as needed inside this function
        :return: the mutated population
        """
        pass

```

### Travelling Salesman Problem

One of the possible applications of genetic algorithms is to the 
[Travelling Salesman Problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem). This problem is NP hard,
meaning that no solution can be achieved in polynomial time, and as such, solutions to most formulations of 
this problem have to be approximated, which makes this problem a good candidate for being solved through
genetic algorithm optimization techniques.

As this particular problem has its own sets of constraints, the base `ContinuousGenAlgSolver` class has to 
be adjusted accordingly. This was done in `TravellingSalesmanProblemSolver`, which can be used out of the box to virtually
all problems of this kind. As an input, this class requires the user to provide a 
[networkx](https://networkx.github.io/documentation/stable/) undirected graph containing all the nodes
representing the problem and with each node connected to every other node by edges with the respective 
associated cost (weight).

```python
from geneal.applications.tsp.travelling_salesman_problem import TravellingSalesmanProblemSolver
from geneal.applications.tsp.examples.world_capitals import G

tsp_solver = TravellingSalesmanProblemSolver(
    graph=G,
    n_genes=241, 
)
```

Again, one can control the solver parameters as before, by providing them on the initialization:

```python
from geneal.applications.tsp.travelling_salesman_problem import TravellingSalesmanProblemSolver
from geneal.applications.tsp.examples.world_capitals import G

tsp_solver = TravellingSalesmanProblemSolver(
    graph=G,
    n_genes=241,
    pop_size=100,
    max_gen=500,
    mutation_rate=0.99,
    selection_rate=0.1 
)
```