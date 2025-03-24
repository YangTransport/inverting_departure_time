import numpy as np
from scipy.integrate import trapezoid

def arrival_time(start_time, queue_length, x, vx):
    start_index = np.abs(x - start_time).argmin(axis=0)
    i = start_index
    res = 0
    while (res < queue_length and i < len(x) - 1):
        res = trapezoid(vx[start_index:i], x[start_index:i], axis=0)
        i += 1
    return x[i]
