import functools
from time import perf_counter


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0]  # this is 'self'
        start = perf_counter()
        value = func(*args, **kwargs)
        run_time = perf_counter() - start
        if hasattr(instance, "tb_writer") and instance.tb_writer is not None:
            instance.log_scalar("time/{}".format(func.__name__), run_time)
        return value

    return wrapper


def stdout_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        instance = args[0]  # this is 'self'
        start = perf_counter()
        value = func(*args, **kwargs)
        run_time = perf_counter() - start
        print(run_time)
        return value

    return wrapper
