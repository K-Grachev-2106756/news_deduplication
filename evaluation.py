from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score


def evaluate_edge_predictions(pred_edges: List[Tuple[int, int]],
                              gt_edges: List[Tuple[int, int]]) -> Dict[str, float]:
    set_gt = set(map(tuple, gt_edges))
    set_preds = set(map(tuple, pred_edges))

    tps = len([i for i in set_gt if i in set_preds or (i[1], i[0]) in set_preds])
    fps = len([i for i in set_preds if i not in set_gt and (i[1], i[0]) not in set_gt])
    fns = len([i for i in set_gt if i not in set_preds and (i[1], i[0]) not in set_preds])

    precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tps,
        'fp': fps,
        'fn': fns
    }


def find_optimal_threshold(predictions: List[float],
                          labels: List[int]) -> Dict[str, float]:
    precisions, recalls, thresholds = precision_recall_curve(labels, predictions)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    return {
        'best_threshold': float(best_threshold),
        'best_f1': float(f1_scores[best_idx]),
        'best_precision': float(precisions[best_idx]),
        'best_recall': float(recalls[best_idx])
    }


def analyze_clusters(clusters_dict: Dict[int, List[int]]) -> Dict[str, any]:
    cluster_sizes = [len(articles) for articles in clusters_dict.values()]
    duplicate_clusters = [size for size in cluster_sizes if size > 1]

    return {
        'total_clusters': len(clusters_dict),
        'duplicate_clusters': len(duplicate_clusters),
        'total_duplicates': sum(duplicate_clusters),
        'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
        'mean_cluster_size': np.mean(duplicate_clusters) if duplicate_clusters else 0.0
    }


def print_evaluation_report(metrics: Dict[str, float]):
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")
    print("="*60 + "\n")
