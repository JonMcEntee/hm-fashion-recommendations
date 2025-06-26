"""
Grid search module for tuning LGBM ranker model hyperparameters.
"""

import pandas as pd
import pickle
from src.models.ranker_model import train_ranker, create_lgbm_recommender
from src.models.generate_recommendations import create_recommendation_generator
from src.features.feature_generator import FeatureGenerator, divide, subtract, Feature, DerivativeFeature
from src.evaluation.metrics import mapk

def run_ranker_grid_search(
    train: pd.DataFrame,
    test: pd.DataFrame,
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
    customers: pd.DataFrame,
    feature_generator,
    param_grid=None,
    last_week=None
):
    if param_grid is None:
        param_grid = {
            'n_estimators': [10, 100, 200, 500],
            'boosting_type': ["gbdt", "dart"],
            'min_child_samples': [50, 75, 100],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    recommender = create_recommendation_generator(transactions, articles)
    results = []
    for n_estimators in param_grid['n_estimators']:
        for boosting_type in param_grid['boosting_type']:
            for min_child_samples in param_grid['min_child_samples']:
                for learning_rate in param_grid['learning_rate']:
                    print(f"Training model with n_estimators={n_estimators}, boosting_type={boosting_type}, min_child_samples={min_child_samples}, learning_rate={learning_rate}")
                    X_train = train.drop(columns=["7d", "35d", "label", 'customer_id', 'article_id', 'prod_name', 'product_group_name'], axis=1)
                    y_train = train["label"]
                    train_baskets = train.groupby(['7d', 'customer_id'], observed=True).size().values
                    model, categorical_cols = train_ranker(
                        X_train,
                        y_train,
                        train_baskets,
                        boosting_type,
                        n_estimators,
                        min_child_samples,
                        learning_rate
                    )
                    lgbm_recommender = create_lgbm_recommender(model, recommender, feature_generator, articles, customers, categorical_cols, default_week=last_week)
                    result = mapk(lgbm_recommender, test, k=12)
                    print(f"MAP@12: {result}")
                    results.append({
                        "n_estimators": n_estimators,
                        "boosting_type": boosting_type,
                        "min_child_samples": min_child_samples,
                        "learning_rate": learning_rate,
                        "mapk": result
                    })
    return results

if __name__ == "__main__":
    print("Loading data...")
    transactions = pd.read_csv("data/transactions_train.csv", parse_dates=['t_dat'])
    articles = pd.read_csv("data/articles.csv")
    customers = pd.read_csv("data/customers.csv")
    ranker_train = pd.read_csv("data/ranker_sample.csv")
    ranker_train = ranker_train.sort_values(["7d", "customer_id"])

    print("Splitting data...")
    last_week = ranker_train["7d"].max() - 1
    train = ranker_train[ranker_train["7d"] < last_week].reset_index(drop=True)
    test = ranker_train[ranker_train["7d"] == last_week].reset_index(drop=True)
    test = test[test.label == 1][['customer_id', 'article_id']]

    print("Loading feature generator...")
    with open('saved_models/feature_generator.pkl', 'rb') as f:
        feature_generator = pickle.load(f)

    print("Starting grid search...")
    results = run_ranker_grid_search(train, test, transactions, articles, customers, feature_generator, last_week=last_week)
    result_df = pd.DataFrame(results).sort_values("mapk", ascending=False)

    print(result_df)
    result_df.to_csv("results/ranker_results.csv", index=False) 