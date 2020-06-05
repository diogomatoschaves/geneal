# geneal

![build_badge](https://github.com/diogomatoschaves/geneal/workflows/build/badge.svg)

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

The above call will perform the optimization with default parameters, which will most likely have to be adjusted for
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
