from copy import deepcopy

import numpy as np
import pandas as pd
from lkmeans.distance import minkowski_distance
from numpy.typing import NDArray


def convert_distance_to_similarity(distance_matrix: NDArray[np.float64], drop_loops: bool = True) -> NDArray[np.float64]:
    matrix = 1/(1 + distance_matrix)
    if drop_loops:
        for i in range(len(matrix)):
            matrix[i][i] = 0
    return matrix


def distances_with_treshold(distance_matrix: NDArray[np.float64], treshold: float) -> NDArray[np.float64]:
    new_distances = deepcopy(distance_matrix)
    new_distances[new_distances < treshold] = 0
    return new_distances


def minkowski_distance_pairwice(data: pd.DataFrame, p: float) ->  NDArray[np.float64]:
    len_data = len(data)
    data = data.to_numpy()
    result = np.zeros((len_data, len_data))
    for i in range(len_data):
        for j in range(len_data):
            result[i][j] = minkowski_distance(data[i], data[j], p)
    return result
