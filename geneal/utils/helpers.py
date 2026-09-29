import numbers

import numpy as np

from geneal.utils.exceptions import InvalidInput


def check_random_state(random_state):
    """
    Turns random_state into the random number generator a solver draws from.

    :param random_state: None, an int seed or a np.random.RandomState
    :return: np.random.RandomState(random_state) for an int, the instance itself
        for a RandomState, and numpy's global RandomState (the one np.random.seed
        seeds) for None
    """

    if random_state is None:
        return np.random.mtrand._rand

    if isinstance(random_state, numbers.Integral):
        return np.random.RandomState(random_state)

    if isinstance(random_state, np.random.RandomState):
        return random_state

    raise InvalidInput(
        f"{random_state!r} is not a valid random_state. "
        f"Use None, an int or a np.random.RandomState instance."
    )


def get_input_dimensions(lst, n_dim=0):
    if isinstance(lst, (list, tuple)):
        return get_input_dimensions(lst[0], n_dim + 1) if len(lst) > 0 else 0
    else:
        return n_dim


def get_elapsed_time(start_time, end_time):

    runtime = (end_time - start_time).seconds

    hours, remainder = divmod(runtime, 3600)
    minutes, seconds = divmod(remainder, 60)

    time_str = ""

    if hours:
        time_str += f"{hours} hours, "

    if minutes:
        time_str += f"{minutes} minutes, "

    if seconds:
        time_str += f"{seconds} seconds"

    return time_str
