"""
Previous Purchases Candidate Generator Module.

This module provides a candidate generator that recommends articles based on users'
previous purchase history. It implements a simple but effective approach that
suggests items that users have bought before, prioritizing more recent purchases.

The generator filters transactions within a specified time window and returns
the most recent purchases for each user, up to a specified limit.
"""

from retrieval.candidate_generator import CandidateGenerator
import pandas as pd
from typing import List


class PreviousPurchases(CandidateGenerator):
    """
    A candidate generator that recommends articles based on users' previous purchases.
    
    This generator implements a simple collaborative filtering approach by suggesting
    items that users have purchased before. It considers only transactions within
    a specified time window and prioritizes more recent purchases.
    """
    
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame, window_size: int = 25):
        """
        Initialize the PreviousPurchases candidate generator.
        
        Args:
            transactions (pd.DataFrame): DataFrame containing transaction data with
                                       columns: customer_id, article_id, week, t_dat
            articles (pd.DataFrame): DataFrame containing article metadata
            window_size (int, optional): Number of weeks to look back for previous
                                       purchases. Defaults to 25.
        """
        super().__init__(transactions, articles)
        self.window_size = window_size

    def set_week(self, week: int) -> None:
        """
        Set the target week for candidate generation.
        
        This method validates that the specified week exists in the transaction data
        and stores it for use in the generate method.
        
        Args:
            week (int): The target week for generating candidates
            
        Raises:
            ValueError: If the specified week is not found in the transaction data
        """
        # Validate that the week exists in the transaction data
        if week not in self.transactions['week'].unique():
            raise ValueError(f"Week {week} not found in data")
        
        self.week = week

    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        """
        Generate candidate recommendations based on previous purchases.
        
        This method filters transactions for the specified users, considers only
        purchases within the time window, and returns the k most recent unique
        purchases for each user.
        
        Args:
            users (List[str]): List of user IDs to generate recommendations for
            k (int): Maximum number of recommendations per user
            
        Returns:
            pd.DataFrame: DataFrame with columns ['customer_id', 'article_id']
                         containing the recommended articles for each user
        """
        # Filter transactions for the specified customers
        filtered_history = self.transactions[self.transactions['customer_id'].isin(users)]
        
        # Only include interactions up to the specified week and within the time window
        filtered_history = filtered_history[
            (filtered_history['week'] < self.week) &
            (filtered_history['week'] >= self.week - self.window_size)
        ]
        
        # Get unique customer-article pairs, sorted by date (most recent first)
        # and remove duplicates to ensure each user-article pair appears only once
        filtered_history = filtered_history[['customer_id', 'article_id', 't_dat']]\
            .sort_values(by='t_dat', ascending=False)\
            .drop_duplicates()\
            .copy()
        
        # Group by customer and take the top k most recent purchases for each user
        result = filtered_history[['customer_id', 'article_id']]\
            .groupby('customer_id')\
            .head(k)\
            .reset_index(drop=True)
        
        return result


if __name__ == "__main__":
    """Example usage of the PreviousPurchases candidate generator."""
    from utils.data_load import load_data
    
    print("Loading data...")
    transactions, articles, customers, customer_map, reverse_customer_map = load_data()
    
    print("Creating previous purchases generator...")
    previous_purchases = PreviousPurchases(transactions, articles)
    
    print("Setting target week...")
    previous_purchases.set_week(50)
    
    print("Generating candidates...")
    # Get first 5 customers for demonstration
    sample_customers = transactions["customer_id"].unique()[:5]
    candidates = previous_purchases.generate(sample_customers, k=3)
    
    print("Generated candidates:")
    print(candidates)