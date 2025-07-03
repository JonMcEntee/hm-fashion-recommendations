"""
Recommendation generator module for H&M Fashion Recommendations.

This module provides functions to create and combine various candidate recommenders,
including weekly bestsellers, previous purchases, and same product code recommenders.
It also defines a hybrid generator that aggregates multiple heuristics and models for candidate generation.
"""
#%%
import sys
import os
project_root = "/Users/jonathanmcentee/Documents/GitHub/hm-fashion-recommendations/"
os.chdir(project_root)
#%%
import pandas as pd
from tqdm import tqdm
from typing import List, Callable, Optional
from src.models.als_model import create_als_recommender
from collections import defaultdict
from src.data_load.data_load import load_data

#%%
def create_weekly_bestsellers_recommender(
    transactions: pd.DataFrame,
) -> Callable[[List[str], int, int], pd.DataFrame]:
    """
    Create a recommender that returns the top-k bestselling articles for each week.
    
    This function precomputes the best-selling articles for each week in the training data.
    When called, the returned recommender function provides the top-k bestsellers from the previous week
    for a given list of customers and a specified week.
    
    Args:
        transactions (pd.DataFrame): The training transactions DataFrame. Must contain 'week' and 'article_id' columns.
    
    Returns:
        Callable: Function that returns top-k bestsellers from the previous week for a list of customers and a specified week.
    
    Usage:
        recommender = create_weekly_bestsellers_recommender(train_df)
        recommendations = recommender([customer1, customer2], week=10, k=12)
    """
    transactions = transactions.copy()

    # Precompute best sellers for all weeks upfront
    best_sellers = transactions.groupby("week")['article_id']\
        .value_counts()\
        .groupby("week")\
        .rank(method='dense', ascending=False)
    best_sellers = best_sellers.rename('bestseller_rank').reset_index()
    best_sellers['week'] += 1  # Shift for easy lookup of "last week"
    best_sellers_dict = best_sellers.groupby('week')['article_id'].apply(list).to_dict()

    def weekly_bestsellers(customers: List[str], week: int, k: int = 12) -> pd.DataFrame:
        """
        Recommend the top-k bestselling articles from the previous week for a list of customers.

        Parameters:
            customers (List[str]): List of customer IDs to generate recommendations for.
            week (int): The week number for which to recommend last week's bestsellers.
            k (int): Number of recommendations per customer (default: 12).

        Returns:
            pd.DataFrame: DataFrame with columns:
                - customer_id
                - article_id
            Each customer receives the same top-k bestsellers from the previous week.

        Usage:
            recommender = create_weekly_bestsellers_recommender(train_df)
            recommendations = recommender([customer1, customer2], week=10, k=12)
        """
        recommendations = best_sellers_dict.get(week, [])[:k]
        num_recommendations = len(recommendations)
        return pd.DataFrame({
            "customer_id": [customer for customer in customers for _ in range(num_recommendations)],
            "article_id": recommendations * len(customers)
        })
    
    return weekly_bestsellers

def create_previous_purchases(
    transactions: pd.DataFrame,
) -> Callable[[List[str], int], pd.DataFrame]:
    """
    Create a recommender that returns customer historical item interactions up to a specified week.
    
    This function processes transaction data to create a history-based recommender that retrieves
    all unique items a customer has interacted with up until a specified week. This is useful for
    analyzing customer purchase patterns and creating temporal-based recommendation features.
    
    Args:
        transactions (pd.DataFrame): The training transactions DataFrame. Must contain columns:
            - t_dat: datetime column with transaction dates
            - customer_id: unique identifier for customers
            - article_id: unique identifier for articles
    
    Returns:
        Callable: Function that returns all unique items a customer has interacted with up to a specified week.
    
    Usage:
        recommender = create_previous_purchases(transactions_df)
        customer_history = recommender(['customer1', 'customer2'], week=10)
    """    
    # Create a clean copy with only needed columns to minimize memory usage
    transactions = transactions[['customer_id', 'week', 'article_id', 't_dat']].copy()

    def previous_purchases(customers: List[str], week: int, k: int = 12) -> pd.DataFrame:
        """
        Retrieve the historical item interactions for specified customers up to a given week.

        Parameters:
            customers (List[str]): List of customer IDs to generate history for.
            week (int): The week number for which to retrieve historical interactions.
            k (int): Number of recommendations per customer (default: 12).
                    Note: Currently not used but kept for API consistency.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - customer_id: Customer identifier
                - article_id: Article identifier
                Each row represents a unique customer-article interaction up to
                and including the specified week.

        Usage:
            recommender = create_previous_purchases(train_df)
            customer_history = recommender(['customer1', 'customer2'], week=10)
        """
        # Filter transactions for the specified customers
        filtered_history = transactions[(transactions['customer_id'].isin(customers))]
        
        # Only include interactions up to the specified week
        filtered_history = filtered_history[filtered_history['week'] < week]
        
        # Get unique customer-article pairs and create a clean copy
        filtered_history = filtered_history[['customer_id', 'article_id', 't_dat']]\
            .sort_values(by='t_dat', ascending=False)\
            .drop_duplicates()\
            .copy()
        
        return filtered_history[['customer_id', 'article_id']].groupby('customer_id').head(k)
    
    return previous_purchases


def create_same_product_code(
    transactions: pd.DataFrame,
    articles: pd.DataFrame,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a function that finds all articles sharing the same product code as a customer's previous purchases.

    This function builds an index mapping each article to its product code. The returned function
    takes a DataFrame of previous purchases and returns all articles that share a product code
    with any of those purchases. This is useful for generating recommendations of similar items
    (e.g., different colors or styles of the same product).

    Args:
        articles (pd.DataFrame): DataFrame containing article metadata. Must include columns:
            - article_id: Unique identifier for each article
            - product_code: Product code grouping similar articles

    Returns:
        Callable: Function that returns articles sharing a product code with previous purchases.

    Usage:
        same_code_finder = create_same_product_code(articles_df)
        similar_articles = same_code_finder(previous_purchases_df)
    """    
    # Create a clean copy with only needed columns to minimize memory usage
    transactions = transactions[['customer_id', 'week', 'article_id', 't_dat']].copy()
    
    last_week = transactions['week'].max()
    transactions["time_decay"] = transactions["week"].apply(lambda x: 0.95 ** (last_week - x))
    weighted_transactions = transactions.groupby(['customer_id', 'article_id', 'week'])['time_decay'].agg('sum').reset_index()
    weighted_transactions['cumulative_weight'] = weighted_transactions\
        .sort_values(by='week', ascending=True)\
        .groupby(['customer_id', 'article_id'])["time_decay"]\
        .cumsum()

    # Build an index mapping each article to its product code
    index = articles[['article_id', 'product_code']]\
        .copy()

    article_to_product_code = index.set_index('article_id')['product_code'].to_dict()
    product_code_to_article_ids = index.groupby('product_code')['article_id'].apply(list).to_dict()
    similar_articles = {article_id: product_code_to_article_ids[article_to_product_code[article_id]] for article_id in index['article_id']}
    
    def same_product_code(customers: List[str], week: int, k: int = 12) -> pd.DataFrame:
        """
        Find all articles that share a product code with the given previous purchases.

        Parameters:
            previous_purchases (pd.DataFrame): DataFrame with at least an 'article_id' column,
                representing articles previously purchased by a customer.

        Returns:
            pd.DataFrame: DataFrame of articles that share a product code with any article in previous_purchases.
                Columns:
                    - customer_id: Customer identifier
                    - article_id: Article sharing a product code with previous purchases

        Usage:
            similar_articles = same_code_finder(previous_purchases_df)
        """
        # Filter transactions for the specified customers
        filtered_history = transactions[(transactions['customer_id'].isin(customers))]
        
        # Only include interactions up to the specified week
        filtered_history = filtered_history[filtered_history['week'] < week]
        
        # Get unique customer-article pairs and create a clean copy
        filtered_history = filtered_history[['customer_id', 'article_id', 't_dat']]\
            .sort_values(by='t_dat', ascending=False)\
            .drop_duplicates()\
            .copy()
                
        same_code = filtered_history[['customer_id', 'article_id']].copy()

        same_code["article_id"] = same_code["article_id"].map(similar_articles)

        same_code = same_code.explode("article_id").astype({"article_id": "int32"})

        # Merge with weighted_transactions to add weights and time decay, filling missing values with 0
        same_code = (
            same_code
            .drop_duplicates()
            .merge(weighted_transactions, on=['customer_id', 'article_id'], how='outer')
            .fillna({"time_decay": 0, "cumulative_weight": 0, "week": 0})
        )

        # Filter to only include articles less than the specified week
        same_code = same_code[same_code["week"] < week]

        # Sort by cumulative_weight (descending) and select top-k per customer
        same_code = (
            same_code
            .sort_values(by=['customer_id', 'cumulative_weight'], ascending=False)
            .groupby('customer_id', as_index=False)
            .head(k)
        )

        return same_code[['customer_id', 'article_id']]

    return same_product_code

def create_item_similarity_recommender(
    transactions: pd.DataFrame
) -> Callable[[List[str], int, int], pd.DataFrame]:
    item_user_matrix, user_map, item_map, reverse_user_map, reverse_item_map = \
        create_user_item_matrix(transactions, last_week, train_window, matrix_type="uniform")

    def item_similarity_recommender(previous_purchases: pd.DataFrame, week: int, k: int = 12) -> pd.DataFrame:
        item_ids = previous_purchases['article_id'].unique()
        similar_items = top_k_cosine_similarity(item_user_matrix, item_ids, k=k)
        user_item_pairs = previous_purchases.merge(similar_items, on='article_id', how='left')\
            .sort_values(by=['customer_id', 'similarity'], ascending=False)\
            .groupby('customer_id', as_index=False)\
            .head(k)

        return user_item_pairs[['customer_id', 'article_id']]
    
    return item_similarity_recommender


def create_recommendation_generator(
    transactions: pd.DataFrame,
    articles: pd.DataFrame
) -> Callable[[List[str], int, int], pd.DataFrame]:
    """
    Create a hybrid recommendation generator that combines multiple heuristics and models.

    This function initializes several recommender systems (baseline, temporal, ALS, previous purchases,
    same product code, weekly bestsellers, and time-weighted bestsellers) using the provided transaction and article data. It returns
    a function that, for a given list of customers, week, and k, generates a set of candidate recommendations
    by aggregating the outputs of all these recommenders and removing duplicates.

    Args:
        transactions (pd.DataFrame): Transaction data. Must include columns required by each recommender.
        articles (pd.DataFrame): Article metadata. Must include columns required by each recommender.

    Returns:
        Callable: Function that generates a set of candidate recommendations by aggregating outputs of all recommenders and removing duplicates.

    Usage:
        recommender = create_recommendation_generator(transactions_df, articles_df)
        recommendations = recommender([customer1, customer2], week=10, k=12)
    """
    # Initialize all base recommenders
    previous_purchases = create_previous_purchases(transactions)
    same_product_code = create_same_product_code(articles)
    weekly_bestsellers = create_weekly_bestsellers_recommender(transactions)

    def recommendation_generator(customers: List[str], week: int, k: int = 12) -> pd.DataFrame:
        """
        Generate a set of candidate recommendations for a list of customers by aggregating multiple recommenders.

        Parameters:
            customers (List[str]): List of customer IDs to generate recommendations for.
            week (int): The week number for which to generate recommendations.
            k (int): Number of recommendations per customer for each recommender (default: 12).

        Returns:
            pd.DataFrame: DataFrame with columns:
                - customer_id: Customer ID
                - article_id: Article ID of the recommended item
            The DataFrame contains deduplicated recommendations aggregated from all recommenders.

        Usage:
            recommender = create_recommendation_generator(transactions_df, articles_df)
            recommendations = recommender([customer1, customer2], week=10, k=12)
        """
        # Get previous purchases for the customers up to the given week
        previous_purchases_df = previous_purchases(customers, week, k=k)
        # Aggregate recommendations from all recommenders
        recommendations = pd.concat([
            previous_purchases_df,
            same_product_code(previous_purchases_df),
            weekly_bestsellers(customers, week, k=k),
        ])
        # Remove duplicate recommendations and drop the 'rank' column if present
        recommendations = recommendations.drop_duplicates()
        if 'rank' in recommendations.columns:
            recommendations = recommendations.drop(columns=['rank'], axis=1)
        return recommendations

    return recommendation_generator


def batch_generate_recommendations(
        transactions: pd.DataFrame,
        file_path: str,
        recommender: Callable[[List[str], int, int], pd.DataFrame] = None,
        k: int = 100,
        first_week: int = 50,
        verbose: bool = False,
        to_csv: bool = False
    ) -> pd.DataFrame:
    """
    Generate and save recommendations for each week in batch mode using a given recommender function.

    Iterates over each week in the transaction data, generates recommendations for all customers in that week
    using the provided recommender, and saves the results to a CSV file. The first week's results overwrite the file,
    and subsequent weeks are appended.

    Args:
        transactions (pd.DataFrame): DataFrame containing transaction data with 't_dat' and 'customer_id' columns.
        recommender (Callable): Recommendation function with signature (customers, week, k) -> pd.DataFrame.
        file_path (str): Path to the output CSV file.
        k (int, optional): Number of recommendations per customer. Defaults to 100.
        first_week (int, optional): The first week to start generating recommendations. Defaults to 50.
        verbose (bool, optional): If True, print progress messages. Defaults to False.

    Returns:
        None. Results are saved to file_path.
    """
    transactions = transactions.copy()

    # Iterate over each week in the specified range
    batch_recommendations = None
    for week in tqdm(range(first_week, last_week + 1), disable=not verbose):
        # Get unique customers for the current week
        customers = transactions[transactions['week'] == week]['customer_id'].unique()
        # Generate recommendations for these customers
        recommendations = recommender(customers, week, k)
        recommendations['week'] = week

        # Save recommendations to CSV (overwrite for first week, append for others)
        if week == first_week:
            if to_csv:
                recommendations.to_csv(file_path, index=False)
            else:
                batch_recommendations = recommendations
        else:
            if to_csv:
                recommendations.to_csv(file_path, mode='a', header=False, index=False)
            else:
                recommendations.to_parquet(file_path, mode='a', index=False)

    return batch_recommendations
#%%
if __name__ == "__main__":
    from src.evaluation.metrics import hit_rate

    print("Loading data...")
    transactions, articles, customers, customer_map, reverse_customer_map = load_data()

    print("Generating same product code...")
    same_product_code = create_same_product_code(transactions, articles)
    batch_generate_recommendations(transactions, same_product_code, "data/same_product_code.csv", verbose=True, to_csv=True)

    # print("Loading recommendations...")
    # recommendations = pd.read_csv("data/weekly_bestsellers.csv")
    # print("Calculating hit rate...")
    # df = hit_rate(recommendations, transactions)
    # df.to_csv("results/weekly_bestsellers_hit_rate.csv", index=False)
