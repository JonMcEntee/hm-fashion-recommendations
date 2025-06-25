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


#%%
print("Loading data...")
transactions = pd.read_csv("data/transactions_train.csv", parse_dates=['t_dat'])
articles = pd.read_csv("data/articles.csv")
customers = pd.read_csv("data/customers.csv")

#%%
ranker_train = pd.read_csv("data/ranker_sample.csv")
ranker_train = ranker_train.sort_values(["7d", "customer_id"])

#%%
print("Preprocessing data...")
excluded = ['article_id', 'customer_id', 'prod_name', 'product_group_name']
obj_columns = ranker_train.select_dtypes(include=['object']).columns.tolist()
categorical_cols = [col for col in obj_columns if col not in excluded]

for col in categorical_cols:
  if col not in excluded:
    ranker_train[col] = ranker_train[col].astype('category')


# %%
print("Splitting data...")
last_week = ranker_train["7d"].max()
train = ranker_train[ranker_train["7d"] < last_week].reset_index(drop=True)
test = ranker_train[ranker_train["7d"] == last_week].reset_index(drop=True)

# put test into expected mapk format
test = test[test.label == 1][['customer_id', 'article_id']]

train = train.sort_values(["7d", "customer_id"])
train_baskets = train.groupby(['7d', 'customer_id'], observed=True).size().values

X_train = train.drop(columns=["7d", "35d", "label", 'customer_id', 'article_id', 'prod_name', 'product_group_name'], axis=1)
y_train = train["label"]

#%%
print("Training model...")
model = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    n_estimators=10,
    importance_type='gain',
    verbose=10
)
model.fit(X_train, y_train, group=train_baskets, categorical_feature=categorical_cols)

#%%
print("Loading feature generator...")
with open('saved_models/feature_generator.pkl', 'rb') as f:
    feature_generator = pickle.load(f)

print("Loading base recommender...")
recommender = create_recommendation_generator(transactions, articles)

#%%
def create_lgbm_recommender(
    ranker: LGBMRanker, 
    recommender: Callable, 
    feature_generator: FeatureGenerator, 
    articles: pd.DataFrame, 
    customers: pd.DataFrame, 
    categorical_cols: List[str]
    ):
    
    article_features = articles[['article_id', 'prod_name', 'product_group_name']]
    customer_features = customers[['customer_id', 'age']]

    def lgbm_recommender(customers: List[str], week: int = 104, k: int = 12) -> pd.DataFrame:

        recommendations = recommender(customers, week, k=100)
        recommendations["7d"] = week
        recommendations["35d"] = week // 5
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

#%%
lgbm_recommender = create_lgbm_recommender(model, recommender, feature_generator, articles, customers, categorical_cols)

# %%
print("Evaluating model...")
result = mapk(lgbm_recommender, test, k=12)
print(f"MAP@12: {result}")
# %%
