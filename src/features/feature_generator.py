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

        self.article_to_group = {
            article: group for article, group in zip(articles.article_id, articles.product_group_name)
        }

        self.product_code_map = articles.groupby('product_code')['article_id'].agg(list).to_dict()
        
        self.train_data["product_code"] = self.train_data["article_id"].map(self.article_to_product)
        self.train_data["product_group_name"] = self.train_data["article_id"].map(self.article_to_group)

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
        print(f"features columns: {features.columns}")
        print(f"article_weeks columns: {article_weeks.columns}")
        print(f"customer_rows columns: {customer_rows.columns}")
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

# Test the feature generator
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    directory = "data/"
    articles = pd.read_csv(directory + "articles.csv")
    customers = pd.read_csv(directory + "customers.csv")
    transactions = pd.read_csv(directory + "transactions_sample.csv",
                             parse_dates=['t_dat'])
    
    # Add week column for temporal split
    print("Preparing data...")
    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['week'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7
    
    # Split data into train and validation sets
    print("Splitting data...")
    train = transactions[transactions.week < last_week - 1].reset_index(drop=True)
    validation = transactions[transactions.week == last_week - 1].reset_index(drop=True)
    
    # Initialize feature generator
    print("Initializing feature generator...")
    feature_generator = FeatureGenerator(train, articles, customers)
    
    # Generate features for validation set
    print("Generating features...")
    validation_features = feature_generator.generate_features(validation)
    
    # Print feature summary
    print("\nFeature Summary:")
    print(f"Number of samples: {len(validation_features)}")
    print("\nFeature columns:")
    for col in validation_features.columns:
        print(f"- {col}")
    
    # Print sample of generated features
    print("\nSample of generated features:")
    print(validation_features.head())
    
    # Save features to CSV
    output_path = "data/validation_features.csv"
    print(f"\nSaving features to {output_path}...")
    validation_features.to_csv(output_path, index=False)
    print("Done!") 