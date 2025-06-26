"""
Evaluation metrics module for H&M Fashion Recommendations.

This module provides functions to compute ranking metrics such as average precision at k (APK
 and mean average precision at k (MAPK) for evaluating recommender system performance.
"""
import numpy as np
import pandas as pd
from typing import List, Set, Callable

def apk(actual: List, predicted: List, k: int = 12) -> float:
    """
    Calculate average precision at k (APK).
    
    Args:
        actual (List): List of actual items.
        predicted (List): List of predicted items.
        k (int): Number of recommendations to consider.
        
    Returns:
        float: Average precision score at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]
    
    score = 0.0
    num_hits = 0.0
    
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    if not actual:
        return 0.0
    
    return score / min(len(actual), k)

def mapk(recommender: Callable, test: pd.DataFrame, k: int = 12) -> float:
    """
    Calculate mean average precision at k (MAPK).
    
    Args:
        recommender (Callable): Function that generates recommendations.
        test (pd.DataFrame): Test data DataFrame.
        k (int): Number of recommendations to consider.
        
    Returns:
        float: Mean average precision score at k.
    """
    customers = test['customer_id'].unique()
    
    actuals = (
        test[test['customer_id'].isin(customers)][['customer_id', 'article_id']]
        .groupby('customer_id')['article_id']
        .apply(set)
        .to_dict()
    )
    
    predictions = recommender(customers)
    
    pred_matrix = (
        predictions.pivot(index='customer_id', columns='rank', values='article_id')
        .reindex(customers)
        .values
    )
    
    actuals_list = [actuals.get(cust_id, set()) for cust_id in customers]
    
    hit_matrix = np.zeros_like(pred_matrix, dtype=np.float32)
    for i, row in enumerate(pred_matrix):
        hit_matrix[i] = [item in actuals_list[i] for item in row]
    
    cumsum_hits = np.cumsum(hit_matrix, axis=1)
    inv_ranks = 1 / np.arange(1, k + 1)
    precisions = hit_matrix * (cumsum_hits * inv_ranks)
    
    actuals_length = np.fromiter(
        (len(a) if a else 1 for a in actuals_list),
        dtype=np.int32,
        count=len(actuals_list)
    )
    
    ap_per_customer = precisions.sum(axis=1) / np.minimum(k, actuals_length)
    
    return float(np.mean(ap_per_customer)) 