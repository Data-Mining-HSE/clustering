import math

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.cluster import KMeans


def get_len(vector: tuple[float, int]):
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)


def get_angle(vector_1: tuple[float, int], vector_2: tuple[float, int]):
    return math.acos(
        (vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1]) /
        (get_len(vector_1) * get_len(vector_2)))


def get_best_index_by_inertia(inertias: list[float], cluster_range: NDArray[np.int64]) -> int:
    max_angle = 0
    best_index = 0
    for index in range(len(inertias) - 2):
        vector_1 = inertias[index] - inertias[index+1], cluster_range[index] - cluster_range[index+1]
        vector_2 = inertias[index+1] - inertias[index+2], cluster_range[index+1] - cluster_range[index+2]
        angle = get_angle(vector_1, vector_2)
        if max_angle < angle:
            max_angle = angle
            best_index = index + 1
    return best_index


def elbow_rule(data: NDArray[np.float64], max_cluster: int) -> plt.Figure:
    cluster_range = np.arange(start=1, stop=max_cluster)

    inertias = []
    for num_cluster in cluster_range:
        kmeanModel = KMeans(n_clusters=num_cluster, n_init='auto').fit(data)
        kmeanModel.fit(data)
        inertias.append(kmeanModel.inertia_)

    best_index = get_best_index_by_inertia(inertias, cluster_range)

    figure = plt.gcf()
    axe = plt.axes()
    axe.plot(cluster_range, inertias, 'bx-')
    axe.plot([cluster_range[best_index]], [inertias[best_index]], 'ro')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method')
    return figure
