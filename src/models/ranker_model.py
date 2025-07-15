"""
Ranker model module for H&M Fashion Recommendations.

This module provides functions to train and use a LightGBM-based learning-to-rank model (LGBMRanker) for personalized product recommendations.
It includes utilities for training, generating recommendations, and evaluating model performance.
"""
#%%
import sys
import os
project_root = "/Users/jonathanmcentee/Documents/GitHub/hm-fashion-recommendations/"
os.chdir(project_root)

#%%
import pandas as pd
import numpy as np
import pickle
from lightgbm import LGBMRanker
from typing import List, Callable
from sklearn.metrics import ndcg_score
from evaluation.metrics import mapk
from features.feature_generator import FeatureGenerator, Feature, DerivativeFeature, divide, subtract
from models.generate_recommendations import create_recommendation_generator


def train_ranker(
        X: pd.DataFrame,
        y: pd.Series,
        train_baskets: np.ndarray,
        boosting_type: str,
        n_estimators: int,
        min_child_samples: int,
        learning_rate: float
    ):
    """
    Train a LightGBM ranker (LGBMRanker) using the provided features and labels.

    Args:
        X (pd.DataFrame): Feature matrix for training.
        y (pd.Series): Target labels for ranking.
        train_baskets (array-like): Group sizes for ranking.
        boosting_type (str): Boosting type for LGBMRanker (e.g., 'gbdt', 'dart').
        n_estimators (int): Number of estimators to be included in the ensemble.
        min_child_samples (int): Minimum number of samples per leaf.
        learning_rate (float): Learning rate for boosting.

    Returns:
        Tuple[LGBMRanker, List[str]]: Trained model and list of categorical columns.
    """
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    for col in categorical_cols:
        if col not in excluded:
            X[col] = X[col].astype('category')

    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type=boosting_type,
        n_estimators=n_estimators,
        importance_type='gain',
        min_child_samples=min_child_samples,
        learning_rate=learning_rate,
        verbose=0
    )

    model.fit(X, y, group=train_baskets, categorical_feature=categorical_cols)

    return model, categorical_cols

def create_lgbm_recommender(
    ranker: LGBMRanker, 
    recommender: Callable, 
    feature_generator: FeatureGenerator, 
    articles: pd.DataFrame, 
    customers: pd.DataFrame, 
    categorical_cols: List[str],
    default_week: int = 104
    ):
    """
    Create a LightGBM-based recommender function that scores and ranks candidate recommendations.

    Args:
        ranker (LGBMRanker): Trained LightGBM ranker model.
        recommender (Callable): Base recommender function to generate candidate items.
        feature_generator (FeatureGenerator): Feature generator for candidate features.
        articles (pd.DataFrame): Article metadata.
        customers (pd.DataFrame): Customer metadata.
        categorical_cols (List[str]): List of categorical feature names.
        default_week (int, optional): Default week for recommendations. Defaults to 104.

    Returns:
        Callable: A function that generates ranked recommendations for a list of customers.
    """
    article_features = articles[['article_id', 'prod_name', 'product_group_name']]
    customer_features = customers[['customer_id', 'age']]

    def lgbm_recommender(customers: List[str], week: int = default_week, k: int = 12) -> pd.DataFrame:
        """
        Generate ranked recommendations for a list of customers using the trained LGBM ranker.

        Args:
            customers (List[str]): List of customer IDs to generate recommendations for.
            week (int, optional): Week number for context. Defaults to default_week.
            k (int, optional): Number of recommendations per customer. Defaults to 12.

        Returns:
            pd.DataFrame: DataFrame with columns [customer_id, article_id, rank].
        """
        recommendations = recommender(customers, week, k=100)
        recommendations["7d"] = week - 1
        recommendations["35d"] = week // 5 - 1
        recommendations = recommendations.merge(article_features, on="article_id", how="left")
        recommendations = recommendations.merge(customer_features, on="customer_id", how="left")
        recommendations = feature_generator.transform(recommendations)

        for col in categorical_cols:
            recommendations[col] = pd.Categorical(
                recommendations[col],
                categories=X_train[col].cat.categories
            )
        
        recommendations["scores"] = ranker.predict(recommendations.drop(columns=["7d", "35d", "customer_id", "article_id", "prod_name", "product_group_name"]))
        recommendations['rank'] = recommendations.groupby('customer_id')['scores'].rank(method='first', ascending=False).astype(int)
        recommendations = recommendations.sort_values(['customer_id', 'rank'], ascending=[False, False])

        return recommendations[["customer_id", "article_id", "rank"]]\
            .sort_values(["customer_id", "rank"])\
            .groupby("customer_id")\
            .head(k)
    
    return lgbm_recommender

if __name__ == "__main__":
    print("Loading data...")
    transactions = pd.read_csv("data/transactions_train.csv", parse_dates=['t_dat'])
    articles = pd.read_csv("data/articles.csv")
    customers = pd.read_csv("data/customers.csv")

    ranker_train = pd.read_csv("data/ranker_sample.csv")
    ranker_train = ranker_train.sort_values(["7d", "customer_id"])

    print("Splitting data...")
    # We are hyperparameter tuning, so we don't use the last week for training
    last_week = ranker_train["7d"].max()
    train = ranker_train[ranker_train["7d"] < last_week].reset_index(drop=True)
    test = ranker_train[ranker_train["7d"] == last_week].reset_index(drop=True)

    # put test into expected mapk format
    test = test[test.label == 1][['customer_id', 'article_id']]

    train = train.sort_values(["7d", "customer_id"])
    train_baskets = train.groupby(['7d', 'customer_id'], observed=True).size().values

    X_train = train.drop(columns=["7d", "35d", "label", 'customer_id', 'article_id', 'prod_name', 'product_group_name'], axis=1)
    y_train = train["label"]

    print("Loading feature generator...")
    with open('saved_models/feature_generator.pkl', 'rb') as f:
        feature_generator = pickle.load(f)

    print("Loading base recommender...")
    recommender = create_recommendation_generator(transactions, articles)

    print("Training model...")
    model, categorical_cols = train_ranker(
        X_train,
        y_train,
        train_baskets,
        boosting_type="dart",
        n_estimators=200,
        min_child_samples=75,
        learning_rate=0.05
    )
    
    lgbm_recommender = create_lgbm_recommender(model, recommender, feature_generator, articles, customers, categorical_cols, default_week=last_week)

    print("Evaluating model...")
    result = mapk(lgbm_recommender, test, k=12)
    print(f"MAP@12: {result}")