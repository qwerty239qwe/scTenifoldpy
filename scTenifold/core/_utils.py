from functools import wraps, partial
import time


__all__ = ["cal_fdr", "timer"]


def cal_fdr(p_vals):
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr


def timer(func=None):
    if func is None:
        return partial(timer)

    @wraps(func)
    def _counter(*args, **kwargs):
        if not kwargs.get("verbosity") is None:
            verbosity = kwargs.pop("verbosity")
        else:
            verbosity = 1
        if verbosity >= 1:
            start = time.perf_counter()
        sol = func(*args, **kwargs)
        if verbosity >= 1:
            end = time.perf_counter()
            print(func.__name__, " processing time: ", str(end - start))
        return sol
    return _counter