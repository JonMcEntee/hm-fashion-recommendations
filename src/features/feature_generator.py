#%%
import sys
import os
project_root = "/Users/jonathanmcentee/Documents/GitHub/hm-fashion-recommendations/"
os.chdir(project_root)

#%%
import pandas as pd
from collections import namedtuple
from typing import Callable

#%%
transactions = pd.read_csv("data/transactions_sample.csv", parse_dates=['t_dat'])
articles = pd.read_csv("data/articles.csv")
customers = pd.read_csv("data/customers.csv")

#%%
# chose 35 days as the larger window because it is a multiple of 7.
# This prevents overlap and leakage.
windows = [7, 35]

for window in windows:
    last = (transactions.t_dat.max() - transactions.t_dat.min()).days // window
    transactions[f'{window}d'] = last - (transactions.t_dat.max() - transactions.t_dat).dt.days // window

window_cols = [f'{window}d' for window in windows]

transactions = transactions\
    .merge(articles[['article_id', 'prod_name', 'product_group_name']], on="article_id", how="left")\
    .merge(customers[['customer_id', 'age']], on="customer_id", how="left")

#%%
Feature = namedtuple("Feature", ["agg", "column", "by", "time_col"])
DerivativeFeature = namedtuple("DerivativeFeature", ["feature_1", "feature_2", "function"])

class FeatureGenerator:
    def __init__(self, features: dict[str, Feature], derivative_features: dict[str, DerivativeFeature], derivative_functions: dict[str, Callable]):
        self.features = features
        self.derivative_features = derivative_features
        self.derivative_functions = derivative_functions
        self.feature_dictionary = {}

    def fit(self, transactions: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame):
        
        for feature_name, feature in self.features.items():
            self.feature_dictionary[feature_name] = self.get_transactions_aggs(
            transactions,
            feature_name,
            feature.column,
            feature.time_col,
            feature.by,
            feature.agg
        )
    
    def transform(self, transactions: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame):
        pass

    # Transactions counts
    def get_transactions_aggs(
        self,
        transactions: pd.DataFrame,
        feature_name: str,
        column: str,
        time_col: str,
        by: list[str] = [],
        agg: str = 'count'
        ) -> pd.DataFrame:

        transactions = transactions.copy()

        if agg not in ['count', 'sum', 'mean', 'std', 'min', 'max']:
            raise Exception(f"Invalid aggregation: {agg}")
        
        if column is None and agg != 'count':
            raise Exception("Column must be provided for non-count aggregations")

        if column is None:
            column = "transactions"
            transactions[column] = 1

        column_customer_agg = transactions\
            .groupby([*by, time_col])[column]\
            .agg(agg)\
            .reset_index(name=feature_name)
        
        return column_customer_agg

#%%
transaction_count_groups = ['article_id', 'prod_name', 'product_group_name']

features = {
    "article_sales_by_week" : Feature(column=None, time_col="7d", by=("article_id",), agg="count"),
    "article_sales_by_month" : Feature(column=None, time_col="35d", by=("article_id",), agg="count"),
    "product_sales_by_week" : Feature(column=None, time_col="7d", by=("prod_name",), agg="count"),
    "product_sales_by_month" : Feature(column=None, time_col="35d", by=("prod_name",), agg="count"),
    "group_sales_by_week" : Feature(column=None, time_col="7d", by=("product_group_name",), agg="count"),
    "group_sales_by_month" : Feature(column=None, time_col="35d", by=("product_group_name",), agg="count"),
    "article_purchases_by_week" : Feature(column=None, time_col="7d", by=("article_id", "customer_id"), agg="count"),
    "article_purchases_by_month" : Feature(column=None, time_col="35d", by=("article_id", "customer_id"), agg="count"),
    "product_purchases_by_week" : Feature(column=None, time_col="7d", by=("prod_name", "customer_id"), agg="count"),
    "product_purchases_by_month" : Feature(column=None, time_col="35d", by=("prod_name", "customer_id"), agg="count"),
    "group_purchases_by_week" : Feature(column=None, time_col="7d", by=("product_group_name", "customer_id"), agg="count"),
    "group_purchases_by_month" : Feature(column=None, time_col="35d", by=("product_group_name", "customer_id"), agg="count"),
    "age_mean_by_week" : Feature(column="age", time_col="7d", by=("article_id",), agg="mean"),
    "age_mean_by_month" : Feature(column="age", time_col="35d", by=("article_id",), agg="mean"),
    "age_std_by_week" : Feature(column="age", time_col="7d", by=("article_id",), agg="std"),
    "age_std_by_month" : Feature(column="age", time_col="35d", by=("article_id",), agg="std"),
    "age_min_by_week" : Feature(column="age", time_col="7d", by=("article_id",), agg="min"),
    "age_min_by_month" : Feature(column="age", time_col="35d", by=("article_id",), agg="min"),
    "age_max_by_week" : Feature(column="age", time_col="7d", by=("article_id",), agg="max"),
    "age_max_by_month" : Feature(column="age", time_col="35d", by=("article_id",), agg="max"),
    "price_mean_by_week" : Feature(column="price", time_col="7d", by=("article_id",), agg="mean"),
    "price_mean_by_month" : Feature(column="price", time_col="35d", by=("article_id",), agg="mean"),
    "price_std_by_week" : Feature(column="price", time_col="7d", by=("article_id",), agg="std"),
    "price_std_by_month" : Feature(column="price", time_col="35d", by=("article_id",), agg="std"),
    "price_min_by_week" : Feature(column="price", time_col="7d", by=("article_id",), agg="min"),
    "price_min_by_month" : Feature(column="price", time_col="35d", by=("article_id",), agg="min"),
    "price_max_by_week" : Feature(column="price", time_col="7d", by=("article_id",), agg="max"),
    "customer_price_mean_by_week" : Feature(column="price", time_col="7d", by=("customer_id",), agg="mean"),
    "customer_price_mean_by_month" : Feature(column="price", time_col="35d", by=("customer_id",), agg="mean"),
    "customer_price_std_by_week" : Feature(column="price", time_col="7d", by=("customer_id",), agg="std"),
    "customer_price_std_by_month" : Feature(column="price", time_col="35d", by=("customer_id",), agg="std"),
    "customer_price_min_by_week" : Feature(column="price", time_col="7d", by=("customer_id",), agg="min"),
    "customer_price_min_by_month" : Feature(column="price", time_col="35d", by=("customer_id",), agg="min"),
    "customer_price_max_by_week" : Feature(column="price", time_col="7d", by=("customer_id",), agg="max"),
    "customer_price_max_by_month" : Feature(column="price", time_col="35d", by=("customer_id",), agg="max"),
    "sales_id_mean_by_week" : Feature(column="sales_channel_id", time_col="7d", by=("article_id",), agg="mean"),
    "sales_id_mean_by_month" : Feature(column="sales_channel_id", time_col="35d", by=("article_id",), agg="mean"),
    "sales_id_min_by_week" : Feature(column="sales_channel_id", time_col="7d", by=("article_id",), agg="min"),
    "sales_id_min_by_month" : Feature(column="sales_channel_id", time_col="35d", by=("article_id",), agg="min"),
    "sales_id_max_by_week" : Feature(column="sales_channel_id", time_col="7d", by=("article_id",), agg="max"),
    "sales_id_max_by_month" : Feature(column="sales_channel_id", time_col="35d", by=("article_id",), agg="max"),
}

derivative_functions = {
    "divide": lambda x, y: x / y,
    "subtract": lambda x, y: x - y
}

derivative_features = {
    "article_purchases_percent": DerivativeFeature(
        feature_1="article_purchases_by_week",
        feature_2="article_sales_by_week",
        function="divide"
    ),
    "product_purchases_percent": DerivativeFeature(
        feature_1="product_purchases_by_week",
        feature_2="product_sales_by_week",
        function="divide"
    ),
    "group_purchases_percent": DerivativeFeature(
        feature_1="group_purchases_by_week",
        feature_2="group_sales_by_week",
        function="divide"
    ),
    "distance_from_mean_age_week": DerivativeFeature(
        feature_1="age",
        feature_2="age_mean_by_week",
        function="subtract"
    ),
    "distance_from_mean_age_month": DerivativeFeature(
        feature_1="age",
        feature_2="age_mean_by_month",
        function="subtract"
    ),
    "distance_from_mean_price_week": DerivativeFeature(
        feature_1="price",
        feature_2="price_mean_by_week",
        function="subtract" 
    ),
    "distance_from_mean_price_month": DerivativeFeature(
        feature_1="price",
        feature_2="price_mean_by_month",
        function="subtract"
    )
}

#%%
if __name__ == "__main__":
    feature_generator = FeatureGenerator(features, derivative_features, derivative_functions)
    feature_generator.fit(transactions, articles, customers)
    print(feature_generator.feature_dictionary)