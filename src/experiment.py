
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tabulate import tabulate

from src.clustering import get_agglomerative, get_em, get_kmeans, get_spectral
from src.compare import pairwice_rand_score


def experiment(data: pd.DataFrame, clusters_list: list[int], data_name: str,
               linkage: str, agglomerative_matrix: NDArray[np.float32], similarity_matrix: NDArray[np.float32]):
    print(f'Experiments with {data_name}\n')

    graph = nx.from_numpy_array(similarity_matrix)

    get_clustering_named_list = [
        (f'Agglomerative ({linkage} linkage)', partial(get_agglomerative,
                                                linkage=linkage,
                                                matrix=agglomerative_matrix)
        ),
        ('KMeans', get_kmeans),
        ('Spectral', get_spectral), # Similarity matrix must be passed to fit
        ('EM Gaussian Mixture', get_em),
    ]

    cluster_results_dict = {}
    modularity_cluster_dict = {}

    for n_clusters in clusters_list:
        result_dict = {}
        modularity_dict = {}

        for clustering_name, clustering_getter in get_clustering_named_list:
            clustering = clustering_getter(n_clusters=n_clusters)
            result = clustering.fit_predict(data) if clustering_name != 'Spectral' else clustering.fit_predict(similarity_matrix)
            result_dict[clustering_name] = result

            modularity = nx.community.modularity(graph, convert_clustering_result_to_groups(result))
            modularity_dict[clustering_name] = modularity

        cluster_results_dict[n_clusters] = result_dict
        modularity_cluster_dict[n_clusters] = modularity_dict
    print_meteric(cluster_results_dict, modularity_cluster_dict)


def print_meteric(cluster_results_dict: dict, modularity_cluster_dict: dict) -> None:
    for num_clusters, result_dict in cluster_results_dict.items():
        print(f'Num Clusters {num_clusters}')

        table = pairwice_rand_score(result_dict)
        print(tabulate(table, headers='keys', tablefmt='psql'))

        modularity = modularity_cluster_dict[num_clusters]
        table = pd.DataFrame(modularity, index=['Modularity'], columns=result_dict.keys())
        print(tabulate(table, headers='keys', tablefmt='psql'))
        print('\n')


def convert_clustering_result_to_groups(result = NDArray[np.int64]):
    num_clusters = max(result) + 1
    groups = []
    for cluster in range(num_clusters):
        groups.append(
            set(np.where(result == cluster)[0])
        )
    return groups
