import numpy as np


def step_function(x):
    return np.array(x > 0, dtype=int)
