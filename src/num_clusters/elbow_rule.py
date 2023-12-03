import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def unit_vector(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    return vector / np.linalg.norm(vector)


def angle_between(v1: NDArray[np.float64], v2: NDArray[np.float64]) -> np.float64:
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_vector_from_arrays(array_1: NDArray[np.float64], array_2: NDArray[np.float64],
                           index_1: int, index_2: int) -> tuple[np.float64, np.float64]:
    x = np.abs(array_1[index_1] - array_1[index_2])
    y = np.abs(array_2[index_1] - array_2[index_2])
    return x, y


def get_best_index_by_inertia(inertias: list[float], cluster_range: NDArray[np.int64]) -> int:
    best_angle = np.inf
    best_index = 0
    for index in range(len(inertias) - 2):
        vector_1 = get_vector_from_arrays(inertias, cluster_range, index, index+1)
        vector_2 = get_vector_from_arrays(inertias, cluster_range, index+1, index+2)
        angle = angle_between(vector_1, vector_2)
        if best_angle > angle:
            best_angle = angle
            best_index = index + 1
    return best_index


def elbow_rule(data: NDArray[np.float64], max_cluster: int, find_best: bool = False) -> plt.Figure:
    cluster_range = np.arange(start=1, stop=max_cluster)
    best_index = None

    inertias = []
    for num_cluster in cluster_range:
        kmeanModel = KMeans(n_clusters=num_cluster, n_init='auto').fit(data)
        kmeanModel.fit(data)
        inertias.append(kmeanModel.inertia_)

    if find_best:
        best_index = get_best_index_by_inertia(inertias, cluster_range)

    figure = plt.gcf()
    axe = plt.axes()
    axe.plot(cluster_range, inertias, 'bx-')
    if best_index:
        axe.plot([cluster_range[best_index]], [inertias[best_index]], 'ro')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method')
    return figure
