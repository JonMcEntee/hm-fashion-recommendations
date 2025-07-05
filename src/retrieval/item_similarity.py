from src.retrieval.candidate_generator import CandidateGenerator
from src.retrieval.previous_purchases import PreviousPurchases
from src.models.collaborative_filtering import create_user_item_matrix, top_k_cosine_similarity
import pandas as pd
from typing import List

class ItemSimilarity(CandidateGenerator):
    def __init__(
            self,
            transactions: pd.DataFrame,
            articles: pd.DataFrame,
            train_window: int = 25,
            matrix_type: str = "uniform",
            n_components: int = 1000
        ):
        super().__init__(transactions, articles)
        self.train_window = train_window
        self.matrix_type = matrix_type
        self.n_components = n_components
        self.item_user_matrix = None
        self.user_map = None
        self.item_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None

    def set_week(self, week: int) -> None:
        self.week = week
        relevant_transactions = self.transactions[self.transactions["week"] < self.week]
        self.item_user_matrix, self.user_map, self.item_map, self.reverse_user_map, self.reverse_item_map = \
            create_user_item_matrix(relevant_transactions, self.week, self.train_window, self.matrix_type)
    
    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        if self.context is None:
            raise ValueError("Context must be set before generating candidates")
        
        if "previous_purchases" not in self.context:
            raise ValueError("Previous purchases must be set in context")
    
        previous_purchases = self.context["previous_purchases"]
        previous_purchases = previous_purchases[previous_purchases["customer_id"].isin(users)]
        item_ids = previous_purchases['article_id'].unique()
        similar_items = top_k_cosine_similarity(
            self.item_user_matrix,
            item_ids,
            map_dict=self.item_map,
            reverse_dict=self.reverse_item_map,
            k=k,
            n_components=self.n_components
        )
        user_item_pairs = previous_purchases.merge(similar_items, on='article_id', how='left')\
            .sort_values(by=['customer_id', 'similarity'], ascending=False)\
            .groupby('customer_id', as_index=False)\
            .head(k)

        return user_item_pairs[['customer_id', 'article_id']]
    
if __name__ == "__main__":
    from src.utils.data_load import load_data
    print("Loading data...")
    transactions, articles, _, customer_map, reverse_customer_map = load_data()
    customers = transactions["customer_id"].unique()[:5]
    print("Creating previous purchases...")
    previous_purchases = PreviousPurchases(transactions, articles)
    previous_purchases.set_week(50)
    df = previous_purchases.generate(customers, k=10)
    print("Creating item similarity...")
    item_similarity = ItemSimilarity(transactions, articles)
    item_similarity.set_context({"previous_purchases": df})
    print("Setting week...")
    item_similarity.set_week(50)
    print("Generating candidates...")
    result = item_similarity.generate(customers, k=10)
    print("Result:")
    print(result)