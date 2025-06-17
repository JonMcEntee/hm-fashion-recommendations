import pandas as pd
import numpy as np
from typing import Dict, List
from collections import defaultdict

class FeatureGenerator:
    def __init__(self, train_data: pd.DataFrame, articles: pd.DataFrame, customers: pd.DataFrame):
        """
        Initialize the feature generator.
        
        Args:
            train_data: Training data DataFrame
            articles: Articles metadata DataFrame
            customers: Customers metadata DataFrame
        """
        self.train_data = train_data
        self.articles = articles
        self.customers = customers
        self.feature_cache = {}
        
        # Precompute mappings
        self.article_to_product = {
            article: product for article, product in zip(articles.article_id, articles.product_code)
        }
        self.product_code_map = articles.groupby('product_code')['article_id'].agg(list).to_dict()
        
        # Precompute customer history
        self.customer_history_pairs = defaultdict(list)
        for _, row in train_data.iterrows():
            self.customer_history_pairs[row['customer_id']].append(
                (row['week'], row['article_id'])
            )
        
    def generate_features(self, samples: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features for the samples.
        
        Args:
            samples: DataFrame containing samples to generate features for
            
        Returns:
            DataFrame with generated features
        """
        customer_rows = self._generate_customer_features(samples)
        article_weeks = self._generate_article_features(samples)
        
        features = customer_rows.merge(article_weeks, on=["article_id", "week"], how="left")
        features = self._add_percentage_features(features)
        
        samples = samples.merge(features, on=["customer_id", "article_id", "week"], how="left")
        samples = samples.merge(self.articles, on=["article_id", "product_code", "product_group_name"], how="left")
        samples = samples.merge(self.customers, on=["customer_id", "age"], how="left")
        
        return samples
    
    def _generate_customer_features(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Generate customer-specific features."""
        customer_rows = samples[["article_id", "customer_id", "week"]].drop_duplicates()
        customer_rows = pd.merge(customer_rows, self.customers[["customer_id", "age"]], on="customer_id")
        
        # Add distance from average age
        age_by_article = pd.merge(self.train_data, self.customers, on="customer_id")\
            .groupby(["week", "article_id"])["age"]\
            .mean()\
            .rename("average_age")\
            .reset_index()
        
        average_age_by_article = {}
        for _, row in age_by_article.iterrows():
            average_age_by_article[(row["article_id"], row["week"])] = row["average_age"]
        
        customer_rows["distance_from_average_age"] = customer_rows.apply(
            lambda x: x["age"] - average_age_by_article.get((x["article_id"], x["week"] - 1), np.nan),
            axis=1
        )
        
        # Add time since last purchase
        customer_rows["time_since_last_purchase"] = customer_rows.apply(
            lambda x: self._get_time_since_last_purchase(x["customer_id"], x["week"]),
            axis=1
        )
        
        return customer_rows
    
    def _generate_article_features(self, samples: pd.DataFrame) -> pd.DataFrame:
        """Generate article-specific features."""
        article_weeks = samples[["article_id", "week"]].drop_duplicates()
        article_weeks = pd.merge(
            article_weeks,
            self.articles[["article_id", "product_code", "product_group_name"]],
            on="article_id"
        )
        
        # Add sales features
        for col in ["article_id", "product_code", "product_group_name"]:
            sales_data = self.train_data.groupby([col, "week"]).size().reset_index(name=f"{col}_sales")
            article_weeks = article_weeks.merge(
                sales_data,
                left_on=[col, "week"],
                right_on=[col, "week"],
                how="left"
            )
        
        return article_weeks
    
    def _add_percentage_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add percentage-based features."""
        features["article_percentage"] = features["last_week_article_purchases"] / features["last_week_article_sales"]
        features["product_percentage"] = features["last_week_product_purchases"] / features["last_week_product_sales"]
        features["group_percentage"] = features["last_week_group_purchases"] / features["last_week_group_sales"]
        
        return features
    
    def _get_time_since_last_purchase(self, customer_id: str, week: int) -> float:
        """Calculate time since last purchase for a customer."""
        for purchase in sorted(self.customer_history_pairs[customer_id], reverse=True):
            if purchase[0] < week:
                return week - purchase[0]
        return np.nan 