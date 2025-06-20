#%%
import sys
import os
project_root = "/Users/jonathanmcentee/Documents/GitHub/hm-fashion-recommendations/"
os.chdir(project_root)
#%%
import pandas as pd
from typing import List, Callable, Optional
from src.models.als_model import create_als_recommender
from src.models.baseline_model import create_baseline_recommender, create_temporal_baseline
from collections import defaultdict

#%%
def create_weekly_bestsellers_recommender(
    transactions: pd.DataFrame,
) -> Callable[[List[str], int, int], pd.DataFrame]:
    """
    Create a recommender that returns the top-k bestselling articles for each week.
    
    This function precomputes the best-selling articles for each week in the training data.
    When called, the returned recommender function provides the top-k bestsellers from the previous week
    for a given list of customers and a specified week.
    
    Parameters:
        transactions (pd.DataFrame): The training transactions DataFrame. Must contain 'week' and 'article_id' columns.
    
    Returns:
        Callable[[List[str], int, int], pd.DataFrame]:
            A function (weekly_bestsellers) that takes a list of customer IDs, a week number, and k,
            and returns a DataFrame with columns:
                - customer_id
                - rank
                - recommendation (article_id)
            The recommendations are the top-k bestsellers from the previous week for each customer.
    
    Usage:
        recommender = create_weekly_bestsellers_recommender(train_df)
        recommendations = recommender([customer1, customer2], week=10, k=12)
    """
    transactions = transactions.copy()
    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['week'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7

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
                - customer_id: Customer ID
                - rank: Rank of the recommendation (1 to k)
                - recommendation: Article ID of the recommended item
            Each customer receives the same top-k bestsellers from the previous week.

        Usage:
            recommender = create_weekly_bestsellers_recommender(train_df)
            recommendations = recommender([customer1, customer2], week=10, k=12)
        """
        recommendations = best_sellers_dict.get(week, [])[:k]
        num_recommendations = len(recommendations)
        return pd.DataFrame({
            "customer_id": [customer for customer in customers for _ in range(num_recommendations)],
            "rank": list(range(1, num_recommendations+1)) * len(customers),
            "recommendation": recommendations * len(customers)
        })
    
    return weekly_bestsellers

#%%
def create_customer_item_recommender(
    transactions: pd.DataFrame,
) -> Callable[[List[str], int], pd.DataFrame]:
    """
    Create a recommender that returns customer historical item interactions up to a specified week.
    
    This function processes transaction data to create a history-based recommender that retrieves
    all unique items a customer has interacted with up until a specified week. This is useful for
    analyzing customer purchase patterns and creating temporal-based recommendation features.
    
    Parameters:
        transactions (pd.DataFrame): The training transactions DataFrame. Must contain columns:
            - t_dat: datetime column with transaction dates
            - customer_id: unique identifier for customers
            - article_id: unique identifier for articles
    
    Returns:
        Callable[[List[str], int], pd.DataFrame]:
            A function (customer_item_recommender) that takes a list of customer IDs and a week number,
            and returns a DataFrame with columns:
                - customer_id: Customer identifier
                - article_id: Article identifiers the customer has interacted with
            Each row represents a unique customer-article interaction up to the specified week.
    
    Usage:
        recommender = create_customer_item_recommender(transactions_df)
        customer_history = recommender(['customer1', 'customer2'], week=10)
    """
    # Convert transaction dates to week numbers relative to the last week
    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['week'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7
    
    # Create a clean copy with only needed columns to minimize memory usage
    transactions = transactions[['customer_id', 'week', 'article_id']].copy()

    def customer_item_recommender(customers: List[str], week: int, k: int = 12) -> pd.DataFrame:
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
            recommender = create_customer_item_recommender(train_df)
            customer_history = recommender(['customer1', 'customer2'], week=10)
        """
        # Filter transactions for the specified customers
        filtered_history = transactions[(transactions['customer_id'].isin(customers))]
        
        # Only include interactions up to the specified week
        filtered_history = filtered_history[filtered_history['week'] <= week]
        
        # Get unique customer-article pairs and create a clean copy
        filtered_history = filtered_history[['customer_id', 'article_id']].drop_duplicates().copy()
        
        return filtered_history
    
    return customer_item_recommender


#%%
if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    directory = "data/"
    transactions = pd.read_csv(directory + "transactions_sample.csv", parse_dates=['t_dat'])
    customers = transactions['customer_id'].unique().tolist()[:1000]
    print("Creating heuristic recommender...")
    # recommender = create_customer_item_recommender(transactions)
    print("Generating recommendations for customers...")

# %%
