import numpy as np
from lkmeans import LKMeans
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture


def get_agglomerative(n_clusters: int, linkage: str, matrix: NDArray[np.float64]) -> AgglomerativeClustering:
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        connectivity=matrix
    )


def get_kmeans(n_clusters: int) -> KMeans:
    return KMeans(
        n_clusters=n_clusters,
        n_init='auto'
    )


def get_spectral(n_clusters: int,) -> SpectralClustering:
    return SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed'
    )


def get_em(n_clusters: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=n_clusters,
    )


def get_lkmeans(n_clusters: int, p: float) -> LKMeans:
    return LKMeans(
        n_clusters=n_clusters,
        p = p
    )
