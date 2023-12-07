from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score, rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score


def _make_comparison(result_dict: dict[str, NDArray[np.int32]], func: Callable) -> NDArray[np.float64]:
    result = np.empty(shape=(len(result_dict), len(result_dict)))

    for index_1, result_array_1 in enumerate(result_dict.values()):
        for index_2, result_array_2 in enumerate(result_dict.values()):
            result[index_1][index_2] = func(result_array_1, result_array_2)
    return result


def pairwise_rand_score(result_dict: dict[str, NDArray[np.int32]]) -> NDArray[np.float64]:
    result = _make_comparison(result_dict, rand_score)
    return pd.DataFrame(result, index=result_dict.keys(), columns=result_dict.keys())


def pairwise_adjusted_rand_score(result_dict: dict[str, NDArray[np.int32]]) -> NDArray[np.float64]:
    result = _make_comparison(result_dict, adjusted_rand_score)
    return pd.DataFrame(result, index=result_dict.keys(), columns=result_dict.keys())


def pairwise_ami_score(result_dict: dict[str, NDArray[np.int32]]) -> NDArray[np.float64]:
    result = _make_comparison(result_dict, adjusted_mutual_info_score)
    return pd.DataFrame(result, index=result_dict.keys(), columns=result_dict.keys())
