from retrieval.candidate_generator import CandidateGenerator
from models.collaborative_filtering import create_user_item_matrix
from typing import List
import pandas as pd
import numpy as np

class GraphSearch(CandidateGenerator):
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame, window: int = 25, max_steps: int = 10):
        super().__init__(transactions, articles)
        self.window = window
        self.max_steps = max_steps

    def set_week(self, week: int) -> None:
        self.week = week

        transaction_window = self.transactions[
            (self.transactions['week'] <= self.week) &
            (self.transactions['week'] >= self.week - self.window)
        ]

        user_item_matrix, user_map, item_map, reverse_user_map, reverse_item_map =\
              create_user_item_matrix(transaction_window)
        
        self.user_item_matrix = user_item_matrix
        self.user_map = user_map
        self.item_map = item_map
        self.reverse_user_map = reverse_user_map
        self.reverse_item_map = reverse_item_map

        item_simularity = self.user_item_matrix.T @ self.user_item_matrix
        if (maximum := item_simularity.max()) > 0:
            item_simularity = item_simularity / maximum

        item_graph = (item_simularity > 0).astype(int)

        self.item_connections = \
            item_graph * self.max_steps + \
            item_simularity * self.max_steps

        for i in range(self.max_steps - 1, 0, -1):
            current_similarity = current_similarity @ item_simularity

            if (maximum := current_similarity.max()) > 0:
                current_similarity = current_similarity / maximum
            else:
                break

            new_connections = current_similarity * (self.item_connections == 0)
            new_connections = (new_connections > 0).astype(int)
            self.item_connections +=\
                  new_connections * i + \
                  new_connections * current_similarity



    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        pass
            

                    
    

if __name__ == "__main__":
    from utils.data_load import load_data
    print("Loading data...")
    transactions, articles, _, customer_map, reverse_customer_map =\
          load_data(transactions_path="data/transactions_sample.csv")
    
    customers = transactions[transactions["week"] == 50]["customer_id"].unique()
    print("Creating graph search...")
    graph_search = GraphSearch(transactions, articles, max_steps=10)
    print("Setting week...")
    graph_search.set_week(50)
    print("Generating recommendations...")
    df = graph_search.generate(customers, k=100)
    print(df)