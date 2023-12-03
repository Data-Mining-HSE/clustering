import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def get_best_index_by_silhouette(silhouette_scores: list[float]) -> int:
    best_index = np.argmax(silhouette_scores)
    return best_index


def silhouette_method(data: NDArray[np.float64], max_cluster: int) -> plt.Figure:
    cluster_range = np.arange(start=2, stop=max_cluster)

    silhouette_scores = []
    for num_cluster in cluster_range:
        kmeanModel = KMeans(n_clusters=num_cluster, n_init='auto').fit(data)
        preds = kmeanModel.fit_predict(data)
        score = silhouette_score(data, preds)
        silhouette_scores.append(score)

    best_index = get_best_index_by_silhouette(silhouette_scores)

    figure = plt.gcf()
    axe = plt.axes()
    axe.plot(cluster_range, silhouette_scores, 'bx-')
    axe.plot([cluster_range[best_index]], [silhouette_scores[best_index]], 'ro')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Scores')
    plt.title('The Silhouette Method')
    return figure
