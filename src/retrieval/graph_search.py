from retrieval.candidate_generator import CandidateGenerator
from models.collaborative_filtering import create_user_item_matrix
from typing import List
import pandas as pd
import numpy as np

class GraphSearch(CandidateGenerator):
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame, window: int = 25):
        super().__init__(transactions, articles)
        self.window = window

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

    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        users = [self.user_map[user] for user in users]
        
        user_articles = {
            user_idx : self.user_item_matrix[user_idx, :].nonzero()[1]
            for user_idx in users
        }

        user_direct_neighbors = {
            user_idx : set(user_articles[user_idx])
            for user_idx in users
        }

        user_article_counts = {
            user_idx : 0
            for user_idx in users
        }

        user_article_sets = {
            user_idx : np.array([])
            for user_idx in users
        }

        iteration = 0
        while users and iteration < 10:
            iteration += 1
            for user_idx in users[:5]:
                items = user_articles[user_idx]
                similar_users = self.user_item_matrix[:, items].nonzero()[0]
                similar_items = self.user_item_matrix[similar_users, :].nonzero()[1]
                similar_items = np.setdiff1d(np.unique(similar_items),
                                             user_article_sets[user_idx])
                
                new_count = user_article_counts[user_idx] + len(similar_items)
                if new_count <= k:
                    user_article_sets[user_idx] = np.concatenate([user_article_sets[user_idx],
                                                                  similar_items])
                    user_article_counts[user_idx] = new_count
                else:
                    user_article_sets[user_idx] = np.concatenate([user_article_sets[user_idx],
                                                                  similar_items[:k - user_article_counts[user_idx]]])
                    user_article_counts[user_idx] = k
                    users.remove(user_idx)

        rows = [
            {'customer_id': self.reverse_user_map[user],
             'article_id': self.reverse_item_map[article]}
            for user, articles in user_article_sets.items()
            for article in articles
        ]

        return pd.DataFrame(rows)
            

                    
    

if __name__ == "__main__":
    from utils.data_load import load_data
    print("Loading data...")
    transactions, articles, _, customer_map, reverse_customer_map =\
          load_data()
    
    customers = transactions[transactions["week"] == 50]["customer_id"].unique()
    print("Creating graph search...")
    graph_search = GraphSearch(transactions, articles)
    print("Setting week...")
    graph_search.set_week(50)
    print("Generating recommendations...")
    df = graph_search.generate(customers, k=100)
    print(df)