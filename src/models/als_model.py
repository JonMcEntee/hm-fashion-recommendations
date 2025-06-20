import numpy as np
import scipy.sparse as sp
import pandas as pd
from implicit.als import AlternatingLeastSquares
from typing import Dict, List, Callable

def train_als_model(
    train: pd.DataFrame,
    factors: int,
    iterations: int,
    regularization: float,
    alpha: float,
    random_seed: int
):
    """
    Prepares data and trains an ALS model, returning the model and mappings.
    """
    # Create user and item mappings
    user_ids = train['customer_id'].unique()
    item_ids = train['article_id'].unique()

    user_map = {id_: idx for idx, id_ in enumerate(user_ids)}
    item_map = {id_: idx for idx, id_ in enumerate(item_ids)}
    reverse_item_map = {idx: id_ for id_, idx in item_map.items()}

    # Create user-item matrix
    train = train.copy()
    train['user_idx'] = train['customer_id'].map(user_map)
    train['item_idx'] = train['article_id'].map(item_map)

    last_date = train.t_dat.max()
    train["time_decay"] = train["t_dat"].apply(lambda x : alpha / (1 + (last_date - x).days))
    purchase_counts = train.groupby(['user_idx', 'item_idx'])["time_decay"].sum().reset_index(name='count')

    item_user_matrix = sp.coo_matrix(
        (purchase_counts['count'].astype(np.float32),
            (purchase_counts['user_idx'], purchase_counts['item_idx'])),
        shape=(len(user_map), len(item_map))
    ).tocsr()

    # Train ALS model
    als_model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_seed
    )
    als_model.fit(item_user_matrix)
    return als_model, user_map, item_map, reverse_item_map, item_user_matrix


def create_als_recommender(
    train: pd.DataFrame = None,
    factors: int = 100,
    iterations: int = 15,
    regularization: float = 0.01,
    alpha: float = 40.0,
    random_seed: int = 42,
    baseline: Callable = None,
    als_model: AlternatingLeastSquares = None,
    user_map: Dict[str, int] = None,
    item_map: Dict[str, int] = None,
    reverse_item_map: Dict[int, str] = None,
    item_user_matrix: sp.csr_matrix = None
) -> Callable:
    """
    Create an ALS-based recommender.
    
    Args:
        train: Training data DataFrame
        factors: Number of latent factors
        iterations: Number of training iterations
        regularization: Regularization parameter
        alpha: Confidence scaling factor
        random_seed: Random seed for reproducibility
        baseline: Fallback recommender for cold start
        
    Returns:
        Function that generates recommendations using ALS model
    """

    if als_model is None:
        als_model, user_map, item_map, reverse_item_map, item_user_matrix = train_als_model(
            train,
            factors,
            iterations,
            regularization,
            alpha,
            random_seed
        )

    def als_recommender(customers: List[str], k: int = 12) -> pd.DataFrame:
        """
        Generate recommendations for a list of customers using ALS model.
        
        Args:
            customers: List of customer IDs
            k: Number of recommendations per customer
            
        Returns:
            DataFrame with recommendations
        """
        recommendations = []
        customer_ids = []
        ranks = []

        for customer in customers:
            if customer not in user_map:
                # Use baseline for cold start
                recs = baseline([customer], k=k)['recommendation'].tolist()
            else:
                user_idx = user_map[customer]
                recommended = als_model.recommend(
                    user_idx,
                    item_user_matrix[user_idx],
                    N=k,
                    filter_already_liked_items=False
                )
                recs = [int(reverse_item_map[item_idx]) for item_idx in recommended[0]]

            recommendations.extend(recs)
            customer_ids.extend([customer] * k)
            ranks.extend(range(1, k + 1))

        return pd.DataFrame({
            "customer_id": customer_ids,
            "rank": ranks,
            "recommendation": recommendations
        })

    return als_recommender

# Code for testing the model
if __name__ == "__main__":
    import pickle

    # Import necessary modules    
    from src.models.baseline_model import create_temporal_baseline
    from src.evaluation.metrics import mapk
    
    # Load and prepare data
    print("Loading data...")
    directory = "data/"
    articles = pd.read_csv(directory + "articles.csv")
    customers = pd.read_csv(directory + "customers.csv")
    transactions = pd.read_csv(directory + "transactions_train.csv",
                               parse_dates=['t_dat'])
    
    # Split data into train and test
    print("Splitting data...")
    train_end_date = transactions['t_dat'].max() - pd.Timedelta(days=7)
    train = transactions[transactions['t_dat'] <= train_end_date].reset_index(drop=True)
    test = transactions[transactions['t_dat'] > train_end_date].reset_index(drop=True)
    
    # Create temporal baseline for cold start
    print("Creating temporal baseline...")
    temporal_baseline = create_temporal_baseline(train)
    
    # Create and train ALS recommender
    print("Training ALS recommender...")
    als_model, user_map, item_map, reverse_item_map, item_user_matrix = train_als_model(
        train,
        factors=100,
        iterations=15,
        regularization=0.01,
        alpha=40.0,
        random_seed=42
    )

    als_recommender = create_als_recommender(
        als_model=als_model,
        user_map=user_map,
        item_map=item_map,
        reverse_item_map=reverse_item_map,
        item_user_matrix=item_user_matrix,
        baseline=temporal_baseline
    )
    
    als_variables = {
        "als_model": als_model,
        "user_map": user_map,
        "item_map": item_map,
        "reverse_item_map": reverse_item_map,
        "item_user_matrix": item_user_matrix
    }

    # Pickle the ALS recommender model
    with open("train_als_model_variables.pkl", "wb") as f:
        pickle.dump(als_variables, f)
    print("ALS recommender model has been pickled to als_recommender.pkl")

    # Evaluate on test set
    print("Evaluating model...")
    score = mapk(als_recommender, test)
    print(f"MAP@12 Score: {score:.4f}")