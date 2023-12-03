import numpy as np
from numpy.typing import NDArray


def convert_distance_to_similarity(distance_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return 1/(1 + distance_matrix)
