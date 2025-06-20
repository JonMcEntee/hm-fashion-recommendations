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
def create_previous_purchases(
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
            A function (previous_purchases) that takes a list of customer IDs and a week number,
            and returns a DataFrame with columns:
                - customer_id: Customer identifier
                - article_id: Article identifiers the customer has interacted with
            Each row represents a the unique articles purchased by the customer up to the specified week.
    
    Usage:
        recommender = create_previous_purchases(transactions_df)
        customer_history = recommender(['customer1', 'customer2'], week=10)
    """
    # Convert transaction dates to week numbers relative to the last week
    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['week'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7
    
    # Create a clean copy with only needed columns to minimize memory usage
    transactions = transactions[['customer_id', 'week', 'article_id']].copy()

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
        filtered_history = filtered_history[filtered_history['week'] <= week]
        
        # Get unique customer-article pairs and create a clean copy
        filtered_history = filtered_history[['customer_id', 'article_id']]\
            .drop_duplicates()\
            .copy()\
            .rename(columns={'article_id': 'recommendation'})
        
        return filtered_history
    
    return previous_purchases


def create_same_product_code(
    articles: pd.DataFrame,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    Create a function that finds all articles sharing the same product code as a customer's previous purchases.

    This function builds an index mapping each article to its product code. The returned function
    takes a DataFrame of previous purchases and returns all articles that share a product code
    with any of those purchases. This is useful for generating recommendations of similar items
    (e.g., different colors or styles of the same product).

    Parameters:
        articles (pd.DataFrame): DataFrame containing article metadata. Must include columns:
            - article_id: Unique identifier for each article
            - product_code: Product code grouping similar articles

    Returns:
        Callable[[pd.DataFrame], pd.DataFrame]:
            A function (same_product_code) that takes a DataFrame of previous purchases (with column 'article_id')
            and returns a DataFrame of articles sharing the same product code.

    Usage:
        same_code_finder = create_same_product_code(articles_df)
        similar_articles = same_code_finder(previous_purchases_df)
    """

    # Build an index mapping each article to its product code
    index = articles[['article_id', 'product_code']]\
        .copy()\
        .rename(columns={'article_id': 'recommendation'})

    def same_product_code(previous_purchases: pd.DataFrame) -> pd.DataFrame:
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
        # Merge previous purchases with the index to get their product codes
        previous_purchases = previous_purchases.copy()
        previous_purchases = (
            previous_purchases
            .merge(index, on='recommendation', how='inner') # Add product_code to previous purchases
            .drop(columns=['recommendation'], axis=1)       # Remove article_id, keep product_code
            .drop_duplicates()                          # Remove duplicate product codes
        )
        # Find all articles that share any of these product codes
        same_code = (
            index
            .merge(previous_purchases, on='product_code', how='inner') # Find all articles with these product codes
            .drop(columns=['product_code'], axis=1)                    # Remove product_code for output
        )
        return same_code

    return same_product_code
#%%
def create_recommendation_generator(
    transactions: pd.DataFrame,
    articles: pd.DataFrame
) -> Callable[[List[str], int, int], pd.DataFrame]:
    """
    Create a hybrid recommendation generator that combines multiple heuristics and models.

    This function initializes several recommender systems (baseline, temporal, ALS, previous purchases,
    same product code, and weekly bestsellers) using the provided transaction and article data. It returns
    a function that, for a given list of customers, week, and k, generates a set of candidate recommendations
    by aggregating the outputs of all these recommenders and removing duplicates.

    Parameters:
        transactions (pd.DataFrame): Transaction data. Must include columns required by each recommender.
        articles (pd.DataFrame): Article metadata. Must include columns required by each recommender.

    Returns:
        Callable[[List[str], int, int], pd.DataFrame]:
            A function (recommendation_generator) that takes a list of customer IDs, a week number, and k,
            and returns a DataFrame with columns:
                - customer_id: Customer ID
                - recommendation: Article ID of the recommended item
            The recommendations are aggregated from all underlying recommenders and deduplicated.

    Usage:
        recommender = create_recommendation_generator(transactions_df, articles_df)
        recommendations = recommender([customer1, customer2], week=10, k=12)
    """
    # Initialize all base recommenders
    baseline_recommender = create_baseline_recommender(transactions)
    temporal_baseline = create_temporal_baseline(transactions)
    als_recommender = create_als_recommender(transactions)
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
                - recommendation: Article ID of the recommended item
            The DataFrame contains deduplicated recommendations aggregated from all recommenders.

        Usage:
            recommender = create_recommendation_generator(transactions_df, articles_df)
            recommendations = recommender([customer1, customer2], week=10, k=12)
        """
        # Get previous purchases for the customers up to the given week
        previous_purchases_df = previous_purchases(customers, week, k=k)
        # Aggregate recommendations from all recommenders
        recommendations = pd.concat([
            baseline_recommender(customers, k=k),
            temporal_baseline(customers, k=k),
            als_recommender(customers, k=k),
            previous_purchases_df,
            same_product_code(previous_purchases_df),
            weekly_bestsellers(customers, week, k=k)
        ])
        # Remove duplicate recommendations and drop the 'rank' column if present
        recommendations = recommendations.drop_duplicates()
        if 'rank' in recommendations.columns:
            recommendations = recommendations.drop(columns=['rank'], axis=1)
        return recommendations

    return recommendation_generator
#%%
if __name__ == "__main__":
    # Example usage
    print("Loading data...")
    directory = "data/"
    transactions = pd.read_csv(directory + "transactions_sample.csv", parse_dates=['t_dat'])
    articles = pd.read_csv(directory + "articles.csv")
    customers = transactions['customer_id'].unique().tolist()[:1000]
    print("Finished loading data")

    recommender = create_recommendation_generator(transactions, articles)
    recommendations = recommender(customers, week=10, k=12)
    print(recommendations)
# %%
