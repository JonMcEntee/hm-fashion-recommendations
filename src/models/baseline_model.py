import pandas as pd
from typing import Callable, List
import numpy as np

def create_baseline_recommender(train: pd.DataFrame) -> Callable:
    """
    Create a baseline recommender based on global popularity.
    
    Args:
        train: Training data DataFrame
        
    Returns:
        Function that generates recommendations based on global popularity
    """
    top = train.groupby("article_id")\
        .size()\
        .sort_values(ascending=False)\
        .reset_index(name="count")\
        .article_id\
        .tolist()
    
    def baseline(customers: List[str], k: int = 12) -> pd.DataFrame:
        """
        Generate recommendations for a list of customers.
        
        Args:
            customers: List of customer IDs
            k: Number of recommendations per customer
            
        Returns:
            DataFrame with recommendations
        """
        return pd.DataFrame({
            "customer_id": [customer for customer in customers for _ in range(k)],
            "rank": list(range(1, k+1)) * len(customers),
            "recommendation": top[:k] * len(customers)
        })
    
    return baseline

def create_temporal_baseline(train: pd.DataFrame) -> Callable:
    """
    Create a temporal baseline recommender based on recent popularity.
    
    Args:
        train: Training data DataFrame
        
    Returns:
        Function that generates recommendations based on recent popularity
    """
    train = train.copy()
    last_date = train.t_dat.max()
    train["time_decay"] = train["t_dat"].apply(lambda x : 1 / (1 + (last_date - x).days))

    top_temporal = train.groupby("article_id")["time_decay"]\
        .sum()\
        .sort_values(ascending=False)\
        .reset_index(name="count")\
        .article_id\
        .tolist()
    
    def temporal_baseline(customers: List[str], k: int = 12) -> pd.DataFrame:
        """
        Generate recommendations for a list of customers.
        
        Args:
            customers: List of customer IDs
            k: Number of recommendations per customer
            
        Returns:
            DataFrame with recommendations
        """
        return pd.DataFrame({
            "customer_id": [customer for customer in customers for _ in range(k)],
            "rank": list(range(1, k+1)) * len(customers),
            "recommendation": top_temporal[:k] * len(customers)
        })
    
    return temporal_baseline

def create_random_baseline(train: pd.DataFrame) -> Callable:
    """
    Create a random baseline recommender.
    
    Args:
        train: Training data DataFrame
        
    Returns:
        Function that generates random recommendations
    """
    # Get unique article IDs
    articles = train['article_id'].unique()
    
    def random_baseline(customers: List[str], k: int = 12) -> pd.DataFrame:
        """
        Generate random recommendations for a list of customers.
        
        Args:
            customers: List of customer IDs
            k: Number of recommendations per customer
            
        Returns:
            DataFrame with random recommendations
        """
        # Generate random recommendations for each customer
        recommendations = []
        for _ in range(len(customers)):
            # Randomly sample k articles without replacement
            recs = np.random.choice(articles, size=k, replace=False)
            recommendations.extend(recs)
        
        return pd.DataFrame({
            "customer_id": [customer for customer in customers for _ in range(k)],
            "rank": list(range(1, k+1)) * len(customers),
            "recommendation": recommendations
        })
    
    return random_baseline 