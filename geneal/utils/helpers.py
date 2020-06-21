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
