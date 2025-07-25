"""
Feature generation module for H&M Fashion Recommendations.

This module defines the FeatureGenerator class and related utilities for creating, fitting,
and transforming features for recommendation models. It supports both basic and derivative features,
and provides aggregation utilities for feature engineering.
"""
#%%
import sys
import os
project_root = "/Users/jonathanmcentee/Documents/GitHub/hm-fashion-recommendations/"
os.chdir(project_root)

#%%
import pandas as pd
from collections import namedtuple, defaultdict
from typing import Callable

Feature = namedtuple("Feature", ["agg", "column", "by"])
DerivativeFeature = namedtuple("DerivativeFeature", ["feature_1", "feature_2", "function"])

class FeatureGenerator:
    """
    FeatureGenerator for aggregating and transforming features for recommendation models.

    This class supports fitting on transaction, article, and customer data, and transforming candidate recommendations with both basic and derivative features.
    """
    def __init__(
        self, 
        features: dict[str, Feature],
        derivative_features: dict[str, DerivativeFeature],
        derivative_functions: dict[str, Callable],
    ):
        """
        Initialize the FeatureGenerator.

        Args:
            features (dict): Dictionary of feature definitions.
            derivative_features (dict): Dictionary of derivative feature definitions.
            derivative_functions (dict): Dictionary mapping function names to callables for derivative features.
        """
        self.features = features
        self.derivative_features = derivative_features
        self.derivative_functions = derivative_functions
        self.feature_dictionary = {}

    def fit(
        self,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        customers: pd.DataFrame,
        verbose: bool = False
    ):
        """
        Fit the feature generator by computing aggregations on the provided data.

        Args:
            transactions (pd.DataFrame): Transaction data.
            articles (pd.DataFrame): Article metadata.
            customers (pd.DataFrame): Customer metadata.
            verbose (bool, optional): If True, print progress messages.
        """
        temp = {}
        same_merge_columns = defaultdict(list)
        for feature_name, feature in self.features.items():
            if verbose:
                print(f" Calculating {feature_name}...")
            temp[feature_name] = self.get_transactions_aggs(
                transactions,
                feature_name,
                feature.column,
                feature.by,
                feature.agg
            )
            same_merge_columns[feature.by].append(feature_name)
        # Merge all features by their group-by columns
        for by, columns in same_merge_columns.items():
            df = None
            for column in columns:
                if verbose:
                    print(f" Merging {column}...")
                if df is None:
                    df = temp[column]
                else:
                    df = df.merge(temp[column], on=by, how="left")
            self.feature_dictionary[by] = df
    
    def transform(self, recommendations: pd.DataFrame, verbose: bool = False):
        """
        Transform candidate recommendations by merging in all fitted features and computing derivative features.

        Args:
            recommendations (pd.DataFrame): DataFrame of candidate recommendations.
            verbose (bool, optional): If True, print progress messages.

        Returns:
            pd.DataFrame: Recommendations with added feature columns.
        """
        for by, df in self.feature_dictionary.items():
            if verbose:
                print(f" Merging group {by}...")
            recommendations = recommendations.merge(df, on=by, how="left")
        for feature_name, feature in self.derivative_features.items():
            if verbose:
                print(f" Calculating Derivative Feature {feature_name}...")
            recommendations[feature_name] = self.derivative_functions[feature.function](
                recommendations[feature.feature_1],
                recommendations[feature.feature_2]
            )
        return recommendations

    def get_transactions_aggs(
        self,
        transactions: pd.DataFrame,
        feature_name: str,
        column: str,
        by: list[str] = [],
        agg: str = 'count'
        ) -> pd.DataFrame:
        """
        Compute aggregations on transaction data for feature engineering.

        Args:
            transactions (pd.DataFrame): Transaction data.
            feature_name (str): Name of the feature to create.
            column (str): Column to aggregate (or None for count).
            by (list[str]): Columns to group by.
            agg (str): Aggregation function ('count', 'sum', 'mean', 'std', 'min', 'max').

        Returns:
            pd.DataFrame: Aggregated feature DataFrame.
        """
        transactions = transactions.copy()
        if agg not in ['count', 'sum', 'mean', 'std', 'min', 'max']:
            raise Exception(f"Invalid aggregation: {agg}")
        if column is None and agg != 'count':
            raise Exception("Column must be provided for non-count aggregations")
        if column is None:
            column = "transactions"
            transactions[column] = 1
        column_customer_agg = transactions\
            .groupby(list(by))[column]\
            .agg(agg)\
            .reset_index(name=feature_name)
        return column_customer_agg

#%%
transaction_count_groups = ['article_id', 'prod_name', 'product_group_name']

features = {
    "article_sales_by_week" : Feature(column=None, by=("article_id", "7d"), agg="count"),
    "article_sales_by_month" : Feature(column=None, by=("article_id", "35d"), agg="count"),
    "product_sales_by_week" : Feature(column=None, by=("prod_name", "7d"), agg="count"),
    "product_sales_by_month" : Feature(column=None, by=("prod_name", "35d"), agg="count"),
    "group_sales_by_week" : Feature(column=None, by=("product_group_name", "7d"), agg="count"),
    "group_sales_by_month" : Feature(column=None, by=("product_group_name", "35d"), agg="count"),
    "article_purchases_by_week" : Feature(column=None, by=("article_id", "customer_id", "7d"), agg="count"),
    "article_purchases_by_month" : Feature(column=None, by=("article_id", "customer_id", "35d"), agg="count"),
    "product_purchases_by_week" : Feature(column=None, by=("prod_name", "customer_id", "7d"), agg="count"),
    "product_purchases_by_month" : Feature(column=None, by=("prod_name", "customer_id", "35d"), agg="count"),
    "group_purchases_by_week" : Feature(column=None, by=("product_group_name", "customer_id", "7d"), agg="count"),
    "group_purchases_by_month" : Feature(column=None, by=("product_group_name", "customer_id", "35d"), agg="count"),
    "age_mean_by_week" : Feature(column="age", by=("article_id", "7d"), agg="mean"),
    "age_mean_by_month" : Feature(column="age", by=("article_id", "35d"), agg="mean"),
    "age_min_by_week" : Feature(column="age", by=("article_id", "7d"), agg="min"),
    "age_min_by_month" : Feature(column="age", by=("article_id", "35d"), agg="min"),
    "age_max_by_week" : Feature(column="age", by=("article_id", "7d"), agg="max"),
    "age_max_by_month" : Feature(column="age", by=("article_id", "35d"), agg="max"),
    "sales_id_mean_by_week" : Feature(column="sales_channel_id", by=("article_id", "7d"), agg="mean"),
    "sales_id_mean_by_month" : Feature(column="sales_channel_id", by=("article_id", "35d"), agg="mean"),
    "sales_id_min_by_week" : Feature(column="sales_channel_id", by=("article_id", "7d"), agg="min"),
    "sales_id_min_by_month" : Feature(column="sales_channel_id", by=("article_id", "35d"), agg="min"),
    "sales_id_max_by_week" : Feature(column="sales_channel_id",  by=("article_id", "7d"), agg="max"),
    "sales_id_max_by_month" : Feature(column="sales_channel_id", by=("article_id", "35d"), agg="max"),
}

def divide(x, y):
    """
    Division of two features.
    """
    return x / y

def subtract(x, y):
    """
    Difference of two features.
    """
    return x - y

derivative_functions = {
    "divide": divide,
    "subtract": subtract
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
    )
}

#%%
if __name__ == "__main__":
    import pickle
    
    print("Loading data...")
    transactions = pd.read_csv("data/transactions_train.csv", parse_dates=['t_dat'])
    articles = pd.read_csv("data/articles.csv")
    customers = pd.read_csv("data/customers.csv")

    print("Calculating windows...")
    # chose 35 days as the larger (monthly) window because it is a multiple of 7.
    # This prevents overlap and leakage.
    windows = [7, 35]

    for window in windows:
        last = (transactions.t_dat.max() - transactions.t_dat.min()).days // window
        transactions[f'{window}d'] = last - (transactions.t_dat.max() - transactions.t_dat).dt.days // window

    window_table = transactions[[f"{window}d" for window in windows]].drop_duplicates()

    print("Merging articles.csv and customers.csv features...")
    transactions = transactions\
        .merge(articles[['article_id', 'prod_name', 'product_group_name']], on="article_id", how="left")\
        .merge(customers[['customer_id', 'age']], on="customer_id", how="left")

    # print("Initializing feature generator...")
    # feature_generator = FeatureGenerator(features, derivative_features, derivative_functions)
    # print("Fitting feature generator...")
    # feature_generator.fit(transactions, articles, customers, verbose=True)

    feature_generator = pickle.load(open('saved_models/feature_generator.pkl', 'rb'))

    # import pickle
    # with open('saved_models/feature_generator.pkl', 'wb') as f:
    #     pickle.dump(feature_generator, f)

    import os

    batch_size = 200_000  # Adjust as needed for memory
    input_path = "data/labeled_recommendations.csv"
    output_path = "data/ranker_train.csv"

    # Remove output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)

    first = True
    reader = pd.read_csv(input_path, chunksize=batch_size)
    for i, samples in enumerate(reader):
        print(f"Processing batch {i+1}...")

        # We want to predict the next week's purchases, so we need to shift the week by 1
        samples['7d'] -= 1

        samples = samples.merge(window_table, on='7d', how='left')
        samples = samples.merge(articles[['article_id', 'prod_name', 'product_group_name']], on='article_id', how='left')
        samples = samples.merge(customers[['customer_id', 'age']], on='customer_id', how='left')

        print("Transforming recommendations...")
        featured_recommendations = feature_generator.transform(samples, verbose=True)

        featured_recommendations['7d'] += 1

        featured_recommendations = featured_recommendations.drop(columns=['35d'], axis=1)
        featured_recommendations = featured_recommendations.merge(window_table, on='7d', how='left')

        # Write to CSV in append mode, header only for first batch
        featured_recommendations.to_csv(
            output_path,
            mode='w' if first else 'a',
            index=False,
            header=first
        )
        first = False

    print(f"Done. Output written to {output_path}")
# %%
