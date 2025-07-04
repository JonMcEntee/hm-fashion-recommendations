from src.retrieval.candidate_generator import CandidateGenerator
import pandas as pd
from typing import List

class WeeklyBestsellers(CandidateGenerator):
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame):
        super().__init__(transactions, articles)

        # Precompute best sellers for all weeks upfront
        best_sellers = self.transactions.groupby("week")['article_id']\
            .value_counts()\
            .groupby("week")\
            .rank(method='dense', ascending=False)
        best_sellers = best_sellers.rename('bestseller_rank').reset_index()
        best_sellers['week'] += 1  # Shift for easy lookup of "last week"
        best_sellers_dict = best_sellers.groupby('week')['article_id'].apply(list).to_dict()

        self.best_sellers_dict = best_sellers_dict

    def set_week(self, week: int) -> None:
        if week not in self.best_sellers_dict:
            raise ValueError(f"Week {week} not found in data")

        self.week = week

    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        recommendations = self.best_sellers_dict.get(self.week, [])[:k]
        num_recommendations = len(recommendations)
        return pd.DataFrame({
            "customer_id": [user for user in users for _ in range(num_recommendations)],
            "article_id": recommendations * len(users)
        })

if __name__ == "__main__":
    from src.utils.data_load import load_data
    print("Loading data...")
    transactions, articles, customers, customer_map, reverse_customer_map = load_data()
    print("Creating weekly bestsellers...")
    weekly_bestsellers = WeeklyBestsellers(transactions, articles)
    print("Setting week...")
    weekly_bestsellers.set_week(50)
    print("Generating candidates...")
    customers = transactions["customer_id"].unique()[:5]
    print("Generating candidates...")
    print(weekly_bestsellers.generate(customers, k=3))