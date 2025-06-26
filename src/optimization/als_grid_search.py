"""
ALS grid search module for H&M Fashion Recommendations.

This module provides a function to perform grid search for hyperparameter tuning of the ALS collaborative filtering model, evaluating each configuration using MAP@12.
"""

import pandas as pd
from typing import List, Dict, Any
from src.models.als_model import create_als_recommender
from src.models.baseline_model import create_temporal_baseline
from src.evaluation.metrics import mapk

def run_als_grid_search(
    fitset: pd.DataFrame,
    validation: pd.DataFrame,
    param_grid: Dict[str, List[Any]] = None
) -> List[Dict[str, Any]]:
    """
    Run grid search to find optimal ALS model hyperparameters.
    
    Args:
        fitset: Training data for model fitting
        validation: Validation data for evaluation
        param_grid: Dictionary of hyperparameters to search over
            Default grid:
            - regularization: [0.1]
            - factors: [20, 30]
            - iterations: [5, 10]
            - alpha: [30, 50, 100, 150]
    
    Returns:
        List of dictionaries containing parameter combinations and their MAP@12 scores
    """
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'regularization': [0.1],
            'factors': [20, 30],
            'iterations': [5, 10],
            'alpha': [30, 50, 100, 150]
        }
    
    # Create temporal baseline for cold start
    temporal_baseline = create_temporal_baseline(fitset)
    
    results = []
    
    # Grid search
    for regularization in param_grid['regularization']:
        for factors in param_grid['factors']:
            for iterations in param_grid['iterations']:
                for alpha in param_grid['alpha']:
                    print(f"\nTraining model with parameters:")
                    print(f"  regularization: {regularization}")
                    print(f"  factors: {factors}")
                    print(f"  iterations: {iterations}")
                    print(f"  alpha: {alpha}")
                    
                    # Create and train ALS recommender
                    als_recommender = create_als_recommender(
                        fitset,
                        factors=factors,
                        iterations=iterations,
                        regularization=regularization,
                        alpha=alpha,
                        random_seed=42,
                        baseline=temporal_baseline
                    )
                    
                    # Evaluate on validation set
                    score = mapk(als_recommender, validation)
                    
                    result = {
                        "regularization": regularization,
                        "factors": factors,
                        "iterations": iterations,
                        "alpha": alpha,
                        "map12": score
                    }
                    results.append(result)
                    print(f"MAP@12 Score: {score:.4f}")
    
    return results

if __name__ == "__main__":
    # Load and prepare data
    print("Loading data...")
    directory = "data/"
    articles = pd.read_csv(directory + "articles.csv")
    customers = pd.read_csv(directory + "customers.csv")
    transactions = pd.read_csv(directory + "transactions_sample.csv",
                             parse_dates=['t_dat'])
    
    # Add week column for temporal split
    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['week'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7
    
    # Split data
    print("Splitting data...")
    fitset = transactions[transactions.week < last_week - 1].reset_index(drop=True)
    validation = transactions[transactions.week == last_week - 1].reset_index(drop=True)
    
    # Run grid search
    print("Starting grid search...")
    results = run_als_grid_search(fitset, validation)
    
    # Print best parameters
    best_result = max(results, key=lambda x: x['map12'])
    print("\nBest parameters:")
    print(f"  regularization: {best_result['regularization']}")
    print(f"  factors: {best_result['factors']}")
    print(f"  iterations: {best_result['iterations']}")
    print(f"  alpha: {best_result['alpha']}")
    print(f"  MAP@12 Score: {best_result['map12']:.4f}") 