from retrieval.candidate_generator import CandidateGenerator
from models.collaborative_filtering import create_user_item_matrix
from typing import List, Tuple
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

class GraphSearch(CandidateGenerator):

    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame, window: int = 25, max_steps: int = 10):
        super().__init__(transactions, articles)
        self.window = window
        self.max_steps = max_steps

    def set_week(self, week: int) -> None:
        self.week = week

        transaction_window = self.transactions[
            (self.transactions['week'] < self.week) &
            (self.transactions['week'] >= self.week - self.window)
        ]

        user_item_matrix, user_map, item_map, reverse_user_map, reverse_item_map =\
              create_user_item_matrix(transaction_window)
        
        self.user_item_matrix = user_item_matrix.tocsr()
        self.user_map = user_map
        self.item_map = item_map
        self.reverse_user_map = reverse_user_map
        self.reverse_item_map = reverse_item_map

        item_simularity = self.user_item_matrix.T @ self.user_item_matrix
        item_graph = (item_simularity > 0).astype(int)

        users, *_ = item_graph.nonzero()
        users = list(users)

        self.item_connections = item_graph
        current_graph = item_graph
        for i in range(self.max_steps - 1, 0, -1):
            print(f"Step {i}... 1")
            current_graph = current_graph @ item_graph

            print(f"Step {i}... 3")
            new_connections = current_graph.tolil()
            print(f"Step {i}... 4")
            new_connections[self.item_connections.nonzero()] = 0
            print(f"Step {i}... 5")
            new_connections.eliminate_zeros()
            print(f"Step {i}... 7")
            new_connections = (new_connections > 0).astype(int)
            print(f"Step {i}... 8")
            self.item_connections += new_connections * i 
            #NOTE: maybe need to use self.item_connections > 0 here?
            current_graph = new_connections
            print(f"Step {i}... Finished!")            

    def _generate(self, users: List[int], k: int) -> pd.DataFrame:
        result = self.context["previous_purchases"].copy()
        result = result[result["customer_id"].isin(users)]

        items = result["article_id"].unique()
        items = [self.item_map[item] for item in items]

        item_values = self.item_connections[items].tocoo()

        df = pd.DataFrame({
            'item_id': [self.reverse_item_map[items[row]] for row in item_values.row],
            'article_id': [self.reverse_item_map[col] for col in item_values.col],
            'score': item_values.data
        })

        result.rename(columns={"article_id": "item_id"}, inplace=True)

        df = pd.merge(result, df, on="item_id", how="left").drop(columns=["item_id"], axis=1)
        df = df.sort_values(by=["customer_id", "score"], ascending=False)
        df.drop(columns=["score"], inplace=True)
        df.drop_duplicates(inplace=True)

        # Remove results in df that are already present in self.context["previous_purchases"]
        prev = self.context["previous_purchases"]
        df = df.merge(prev, on=["customer_id", "article_id"], how="left", indicator=True)
        df = df[df["_merge"] == "left_only"].drop(columns=["_merge"])

        df = df.groupby("customer_id").head(k)

        return df


        

                    
    

if __name__ == "__main__":
    from utils.data_load import load_data
    from retrieval.previous_purchases import PreviousPurchases

    print("Loading data...")
    transactions, articles, _, customer_map, reverse_customer_map =\
          load_data()
    customers = transactions[transactions["week"] == 50]["customer_id"].unique()

    previous_purchases = PreviousPurchases(transactions, articles)
    previous_purchases.set_week(50)

    context = {
        "previous_purchases": previous_purchases.generate(customers, k=10)
    }

    print("Creating graph search...")
    graph_search = GraphSearch(transactions, articles, max_steps=3)
    print("Setting week...")
    graph_search.set_week(50)
    graph_search.set_context(context)
    print("Generating recommendations...")
    df = graph_search.generate(customers, k=100)
    print(df)