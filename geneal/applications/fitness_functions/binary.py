import numpy as np


def fitness_functions_binary(function_number):

    if function_number == 1:

        return lambda chromosome: chromosome.sum()

    elif function_number == 2:

        answer = np.array(
            [
                1,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
            ]
        )

        return lambda chromosome: -np.abs(answer - chromosome).sum()
