#%%
import sys
import os
project_root = "/Users/jonathanmcentee/Documents/GitHub/hm-fashion-recommendations/"
os.chdir(project_root)

#%%
import pandas as pd
import pickle
from lightgbm import LGBMRanker
from typing import List, Callable
from sklearn.metrics import ndcg_score
from src.evaluation.metrics import mapk
from src.features.feature_generator import FeatureGenerator, Feature, DerivativeFeature, divide, subtract
from src.models.generate_recommendations import create_recommendation_generator

def train_ranker(X, y, train_baskets, boosting_type, n_estimators, min_child_samples, learning_rate):
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
    
    article_features = articles[['article_id', 'prod_name', 'product_group_name']]
    customer_features = customers[['customer_id', 'age']]

    def lgbm_recommender(customers: List[str], week: int = default_week, k: int = 12) -> pd.DataFrame:

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
    last_week = ranker_train["7d"].max() - 1
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

    results = []
    for n_estimators in [10, 100, 200, 500]:
        for boosting_type in ["gbdt", "dart"]:
            for min_child_samples in [50, 75, 100]:
                for learning_rate in [0.01, 0.05, 0.1]:
                    print("Training model...")
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

                    print("Evaluating model...")
                    result = mapk(lgbm_recommender, test, k=12)
                    print(f"MAP@12 for {n_estimators} estimators, {boosting_type} boosting, {min_child_samples} min child samples, and learning rate {learning_rate}: {result}")
                    results.append({
                        "n_estimators": n_estimators,
                        "boosting_type": boosting_type,
                        "min_child_samples": min_child_samples,
                        "learning_rate": learning_rate,
                        "mapk": result
                    })

    result_df = pd.DataFrame(results).sort_values("mapk", ascending=False)
    print(result_df)
    result_df.to_csv("results/ranker_results.csv", index=False)