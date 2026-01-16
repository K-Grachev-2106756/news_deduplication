from typing import List, Tuple, Dict
import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from collections import defaultdict


def clusters_from_edges(edges: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    G = nx.Graph()
    G.add_edges_from(edges)
    sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    clusters_dict = {}
    for i, subgraph in enumerate(sub_graphs):
        clusters_dict[i] = list(subgraph.nodes())

    return clusters_dict


def louvain_clustering(edges: List[Tuple[int, int]], resolution: float = 1.0) -> Dict[int, List[int]]:
    G = nx.Graph()
    G.add_edges_from(edges)
    communities = nx_comm.louvain_communities(G, resolution=resolution)

    clusters_dict = {}
    for i, community in enumerate(communities):
        clusters_dict[i] = list(community)

    return clusters_dict


def clusters_dict_to_labels(clusters_dict: Dict[int, List[int]], n_articles: int) -> np.ndarray:
    labels = np.full(n_articles, -1)
    for cluster_id, article_indices in clusters_dict.items():
        for idx in article_indices:
            labels[idx] = cluster_id
    return labels


def unionfind_clustering(edges: List[Tuple[int, int]], n_articles: int) -> Dict[int, List[int]]:
    from unionfind import unionfind

    uf = unionfind(n_articles)

    for idx1, idx2 in edges:
        uf.unite(idx1, idx2)

    index_to_cluster = {}
    clusters_dict = defaultdict(list)

    for idx in range(n_articles):
        root = uf.find(idx)
        if root not in index_to_cluster:
            index_to_cluster[root] = len(index_to_cluster)
        cluster_id = index_to_cluster[root]
        clusters_dict[cluster_id].append(idx)

    return dict(clusters_dict)
