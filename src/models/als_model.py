import numpy as np
import scipy.sparse as sp
import pandas as pd
from implicit.als import AlternatingLeastSquares
from typing import Tuple, Dict, List, Callable

def train_als_model(
    train: pd.DataFrame,
    factors: int,
    iterations: int,
    regularization: float,
    alpha: float,
    random_seed: int
) -> Tuple[AlternatingLeastSquares, sp.csr_matrix, Dict, Dict]:
    """
    Train ALS model on the training data.
    
    Args:
        train: Training data DataFrame
        factors: Number of latent factors
        iterations: Number of training iterations
        regularization: Regularization parameter
        alpha: Confidence scaling factor
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (trained model, user-item matrix, user mapping, item mapping)
    """
    user_ids = train['customer_id'].unique()
    item_ids = train['article_id'].unique()

    user_map = {id_: idx for idx, id_ in enumerate(user_ids)}
    item_map = {id_: idx for idx, id_ in enumerate(item_ids)}

    train['user_idx'] = train['customer_id'].map(user_map)
    train['item_idx'] = train['article_id'].map(item_map)

    purchase_counts = train.groupby(['user_idx', 'item_idx'])["time_decay"].sum().reset_index(name='count')

    item_user_matrix = sp.coo_matrix(
        (purchase_counts['count'].astype(np.float32),
         (purchase_counts['user_idx'], purchase_counts['item_idx'])),
        shape=(len(user_map), len(item_map))
    ).tocsr()

    als_model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_seed
    )

    als_model.fit(item_user_matrix)

    def als_recommender(
    customer: str,
    als_model: AlternatingLeastSquares,
    item_user_matrix: sp.csr_matrix,
    user_map: Dict,
    item_map: Dict,
    temporal_baseline: Callable,
    k: int = 12
) -> List[int]:
    """
    Generate recommendations for a customer using ALS model.
    
    Args:
        customer: Customer ID
        als_model: Trained ALS model
        item_user_matrix: User-item interaction matrix
        user_map: Mapping from customer IDs to matrix indices
        item_map: Mapping from article IDs to matrix indices
        temporal_baseline: Fallback recommender for cold start
        k: Number of recommendations to generate
        
    Returns:
        List of recommended article IDs
    """
    if customer not in user_map:
        return temporal_baseline(customer, k=k)

    reverse_item_map = {idx: id_ for id_, idx in item_map.items()}
    user_idx = user_map[customer]
    recommended = als_model.recommend(
        user_idx,
        item_user_matrix[user_idx],
        N=k,
        filter_already_liked_items=False
    )

    return [int(reverse_item_map[item_idx]) for item_idx in recommended[0]] 

    return als_model, item_user_matrix, user_map, item_map