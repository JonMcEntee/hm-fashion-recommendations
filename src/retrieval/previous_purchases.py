from src.retrieval.candidate_generator import CandidateGenerator
import pandas as pd
from typing import List

class PreviousPurchases(CandidateGenerator):
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame, window_size: int = 25):
        super().__init__(transactions, articles)
        self.window_size = window_size

    def set_week(self, week: int) -> None:
        self.week = week

    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        # Filter transactions for the specified customers
        filtered_history = self.transactions[self.transactions['customer_id'].isin(users)]
        
        # Only include interactions up to the specified week
        filtered_history = filtered_history[
            (filtered_history['week'] < self.week) &\
            (filtered_history['week'] >= self.week - self.window_size)
        ]
        
        # Get unique customer-article pairs and create a clean copy
        filtered_history = filtered_history[['customer_id', 'article_id', 't_dat']]\
            .sort_values(by='t_dat', ascending=False)\
            .drop_duplicates()\
            .copy()
        
        result = filtered_history[['customer_id', 'article_id']]\
            .groupby('customer_id')\
            .head(k)\
            .reset_index(drop=True)
        
        return result

if __name__ == "__main__":
    from src.utils.data_load import load_data
    print("Loading data...")
    transactions, articles, customers, customer_map, reverse_customer_map = load_data()
    print("Creating previous purchases...")
    previous_purchases = PreviousPurchases(transactions, articles)
    print("Setting week...")
    previous_purchases.set_week(50)
    print("Generating candidates...")
    customers = transactions["customer_id"].unique()[:5]
    print("Generating candidates...")
    candidates = previous_purchases.generate(customers, k=3)
    print(candidates)