import math

import numpy as np


def norm_2(arr_1, arr_2):
    assert len(arr_1) == len(arr_2)
    result = 0.0
    for i in range(len(arr_1)):
        result += (arr_1[i] - arr_2[i]) ** 2
    return math.sqrt(result)

def KL(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))