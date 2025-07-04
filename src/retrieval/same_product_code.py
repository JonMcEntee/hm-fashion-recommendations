from src.retrieval.candidate_generator import CandidateGenerator
import pandas as pd
from typing import List

class SameProductCode(CandidateGenerator):
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame):
        super().__init__(transactions, articles)

        transactions = transactions[['customer_id', 'week', 'article_id', 't_dat']].copy()
    
        last_week = transactions['week'].max()
        transactions["time_decay"] = transactions["week"].apply(lambda x: 0.95 ** (last_week - x))
        weighted_transactions = transactions.groupby(['article_id', 'week'])['time_decay'].agg('sum').reset_index()
        weighted_transactions['cumulative_weight'] = weighted_transactions\
            .sort_values(by='week', ascending=True)\
            .groupby(['article_id'])["time_decay"]\
            .cumsum()
        
        self.weighted_transactions = weighted_transactions

        # Build a DataFrame mapping each article_id to its product_code
        index = articles[['article_id', 'product_code']].copy()

        # Group by product_code to get all article_ids for each product_code
        product_code_to_articles = index.groupby('product_code')['article_id'].apply(list)

        # Map each article_id to its list of similar articles (same product_code)
        index['similar_articles'] = index['product_code'].map(product_code_to_articles)

        # Convert to dict: article_id -> list of similar articles
        self.similar_articles = dict(zip(index['article_id'], index['similar_articles']))


    def set_week(self, week: int) -> None:
        self.week = week
        
    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        if self.context is None:
            raise ValueError("Context must be set before generating candidates")
        
        if "previous_purchases" not in self.context:
            raise ValueError("Previous purchases must be set in context")
                
        same_code = self.context["previous_purchases"][['customer_id', 'article_id']].copy()        
        same_code = same_code[same_code["customer_id"].isin(users)]
        same_code["article_id"] = same_code["article_id"].map(self.similar_articles)
        same_code = same_code.explode("article_id").astype({"article_id": "int32"})

        # Filter to only include weights for articles less than the specified week
        relevant_weights = self.weighted_transactions[self.weighted_transactions["week"] < self.week].copy()

        # Add a column to the weighted_transactions DataFrame that contains the maximum week for each article_id
        relevant_weights["max_week"] = relevant_weights.groupby("article_id")["week"].transform("max")

        # Find most recent week for each article_id
        relevant_weights = relevant_weights[relevant_weights["max_week"] == relevant_weights["week"]]

        # Drop the max_week and week columns
        relevant_weights = relevant_weights.drop(columns=["max_week", "week"])

        same_code = (
            same_code
            .drop_duplicates()
            .merge(relevant_weights, on=['article_id'], how='left')
            .sort_values(by=['customer_id', 'cumulative_weight'], ascending=False)
            .groupby('customer_id', as_index=False)
            .head(k)
            .reset_index(drop=True)
        )

        return same_code[['customer_id', 'article_id']]

if __name__ == "__main__":
    from src.utils.data_load import load_data
    from src.retrieval.previous_purchases import PreviousPurchases
    print("Loading data...")
    transactions, articles, customers, customer_map, reverse_customer_map = load_data()
    customers = transactions["customer_id"].unique()[:5]
    print("Creating previous purchases...")
    previous_purchases = PreviousPurchases(transactions, articles)
    previous_purchases.set_week(50)
    result = previous_purchases.generate(customers, k=10)
    print("Creating same product code...")
    same_product_code = SameProductCode(transactions, articles)
    print("Setting week...")
    same_product_code.set_week(50)
    same_product_code.set_context({"previous_purchases": result})
    print("Generating candidates...")
    candidates = same_product_code.generate(customers, k=3)
    print(candidates)
    print(candidates["customer_id"].value_counts())