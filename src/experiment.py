
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import silhouette_score
from tabulate import tabulate

from src.clustering import (get_agglomerative, get_em, get_kmeans, get_lkmeans,
                            get_spectral)
from src.compare import pairwise_ami_score, pairwise_rand_score


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
        ('LKmeans (4)', partial(get_lkmeans, p=4)),
        ('LKMeans (0.5)', partial(get_lkmeans, p=0.5))
    ]

    cluster_results_dict = {}
    modularity_cluster_dict = {}
    silhouette_cluster_dict = {}

    for n_clusters in clusters_list:
        result_dict = {}
        modularity_dict = {}
        silhouette_dict = {}

        for clustering_name, clustering_getter in get_clustering_named_list:
            clustering = clustering_getter(n_clusters=n_clusters)
            if clustering_name == 'Spectral':
                result = clustering.fit_predict(similarity_matrix)
            elif 'LKmeans'.lower() in clustering_name.lower():
                result = clustering.fit_predict(data.to_numpy())
                result = np.array(result)
            else:
                result = clustering.fit_predict(data)
            result_dict[clustering_name] = result

            modularity = nx.community.modularity(graph, convert_clustering_result_to_groups(result))
            modularity_dict[clustering_name] = modularity

            silhouette_dict[clustering_name] = silhouette_score(data, result)

        cluster_results_dict[n_clusters] = result_dict
        modularity_cluster_dict[n_clusters] = modularity_dict
        silhouette_cluster_dict[n_clusters] = silhouette_dict
    print_meteric(cluster_results_dict, modularity_cluster_dict, silhouette_cluster_dict)


def print_meteric(cluster_results_dict: dict, modularity_cluster_dict: dict, silhouette_cluster_dict: dict) -> None:
    for num_clusters, result_dict in cluster_results_dict.items():
        print(f'Num Clusters {num_clusters}')

        print('Pairwise RI Score')
        table = pairwise_rand_score(result_dict)
        print(tabulate(table, headers='keys', tablefmt='psql'))

        print('Pairwise AMI Score')
        table = pairwise_ami_score(result_dict)
        print(tabulate(table, headers='keys', tablefmt='psql'))

        print('Modularity')
        modularity = modularity_cluster_dict[num_clusters]
        table = pd.DataFrame(modularity, index=['Modularity'], columns=result_dict.keys())
        print(tabulate(table, headers='keys', tablefmt='psql'))

        print('Silhouette')
        silhouette = silhouette_cluster_dict[num_clusters]
        table = pd.DataFrame(silhouette, index=['Silhouette'], columns=result_dict.keys())
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
