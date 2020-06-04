import numpy as np


def fitness_functions_continuous(function_number):

    if function_number == 1:
        return lambda chromosome: -(np.abs(chromosome[0]) + np.cos(chromosome[0]))
    elif function_number == 2:
        return lambda chromosome: -(np.abs(chromosome[0]) + np.sin(chromosome[0]))
    elif function_number == 3:
        return lambda chromosome: -(chromosome ** 2).sum()
    elif function_number == 4:
        return lambda chromosome: -np.sum(
            np.abs(chromosome) - 10 * np.cos(np.sqrt(np.abs(10 * chromosome)))
        )
    elif function_number == 5:
        return lambda chromosome: -(chromosome[0] ** 2 + chromosome[0]) * np.cos(
            chromosome[0]
        )
    elif function_number == 6:
        return lambda chromosome: -(
            chromosome[0] * np.sin(4 * chromosome[0])
            + 1.1 * chromosome[1] * np.sin(2 * chromosome[1])
        )
