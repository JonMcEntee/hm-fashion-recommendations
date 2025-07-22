from retrieval.candidate_generator import CandidateGenerator
from models.collaborative_filtering import create_user_item_matrix
from typing import List, Tuple
import pandas as pd
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

class GraphSearch(CandidateGenerator):
    """
    Graph search is a retrieval method that uses a matrix representation of a graph to find similar items.
    """

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

        item_simularity = (self.user_item_matrix.T @ self.user_item_matrix).tocsr()

        users, *_ = item_simularity.nonzero()
        users = list(users)
        print(f"sparisfying matrix...")
        item_simularity, users = self._sparsify_matrix(item_simularity, users, 500)

        print(f"Matrix type:\n{type(item_simularity)}")
        if (maximum := item_simularity.max()) > 0:
            item_simularity = item_simularity / maximum

        item_graph = (item_simularity > 0).astype(int)

        self.item_connections = \
            item_graph * self.max_steps + \
            item_simularity * self.max_steps

        print(f"item_similarity shape: {item_simularity.shape}")
        print(f"item_similarity nnz: {item_simularity.nnz}")
        print(f"item_similarity density: {item_simularity.nnz / (item_simularity.shape[0] * item_simularity.shape[1]):.6f}")


        current_similarity = item_simularity
        for i in range(self.max_steps - 1, 0, -1):
            print(f"Step {i}... 1")
            current_similarity = (current_similarity @ item_simularity).tocsr()
            current_similarity, users = self._sparsify_matrix(current_similarity, users, 500)

            print(f"Step {i}... 2")

            if (maximum := current_similarity.max()) > 0:
                current_similarity = current_similarity / maximum
            else:
                break

            print(f"Step {i}... 3")
            new_connections = current_similarity.copy().tolil()
            print(f"Step {i}... 4")
            new_connections[self.item_connections.nonzero()] = 0
            print(f"Step {i}... 5")
            new_connections = new_connections.tocsr()
            print(f"Step {i}... 6")
            new_connections.eliminate_zeros()
            print(f"Step {i}... 7")
            new_connections = (new_connections > 0).astype(int)
            print(f"Step {i}... 8")
            self.item_connections +=\
                  new_connections * i + \
                  new_connections * current_similarity
            print(f"Step {i}... Finished!")
        
    def _sparsify_matrix(
            self,
            matrix: sp.csr_matrix,
            active_users: List[int],
            k: int
        ) -> Tuple[sp.csr_matrix, List[int]]:

        new_matrix = sp.lil_matrix(matrix.shape, dtype=matrix.dtype)
        
        for user in active_users:
            start = matrix.indptr[user]
            end = matrix.indptr[user + 1]

            if end > start:
                row_data = matrix.data[start:end]
                row_indices = matrix.indices[start:end]

                if len(row_data) > k:
                    top_k_indices = np.argsort(row_data)[-k:]
                    row_data = row_data[top_k_indices]
                    row_indices = row_indices[top_k_indices]
                    active_users.remove(user)
                
                new_matrix[user, row_indices] = row_data
        
        return new_matrix.tocsr(), active_users

            

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