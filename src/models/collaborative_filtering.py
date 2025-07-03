import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, Dict, List
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

def create_user_item_matrix(
    train: pd.DataFrame,
    last_week: int,
    train_window: int,
    matrix_type: str = "uniform"
) -> Tuple[sp.csr_matrix, Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    # Filter transactions to only include those within the specified training window
    train = train[train['week'] <= last_week]
    train = train[train['week'] >= last_week - train_window]

    # Create unique user and item lists
    user_ids = train['customer_id'].unique()
    item_ids = train['article_id'].unique()

    # Create mappings between ids and matrix indices
    user_map = {id_: idx for idx, id_ in enumerate(user_ids)}
    item_map = {id_: idx for idx, id_ in enumerate(item_ids)}
    reverse_user_map = {idx: id_ for id_, idx in user_map.items()}
    reverse_item_map = {idx: id_ for id_, idx in item_map.items()}

    # Map customer and article ids to their matrix indices
    train = train.copy()
    train['user_idx'] = train['customer_id'].map(user_map)
    train['item_idx'] = train['article_id'].map(item_map)

    if matrix_type == "time_decay":
        # Weight interactions by recency (more recent = higher weight)
        last_date = train.t_dat.max()
        train["time_decay"] = train["t_dat"].apply(lambda x : (last_date - x).days)
        associations = train.groupby(['user_idx', 'item_idx'])["time_decay"].sum().reset_index(name='count')
    elif matrix_type == "uniform":
        # Each user-item interaction is counted once (binary)
        associations = train[['user_idx', 'item_idx']].drop_duplicates()
        associations['count'] = 1

    # Build the sparse user-item matrix
    item_user_matrix = sp.coo_matrix(
        (associations['count'].astype(np.float32),
        (associations['user_idx'], associations['item_idx'])),
        shape=(len(user_map), len(item_map))
    ).tocsr()

    return item_user_matrix, user_map, item_map, reverse_user_map, reverse_item_map

def top_k_cosine_similarity(
    item_user_matrix: sp.csr_matrix,
    ids: List[int],
    map_dict: Dict[int, str],
    reverse_dict: Dict[int, str],
    type: str = "item", # "item" or "user"
    k: int = 20,
    n_components: int = 1000,
    random_state: int = None,
    verbose: bool = False
):
    k += 1 # Add 1 to k to make room for the object itself (we will remove it later)

    if type not in ["item", "user"]:
        raise ValueError("Type must be either 'item' or 'user'")

    transformed_ids = [map_dict[ob] for ob in ids if ob in map_dict]

    if type == "item":
        item_user_matrix = item_user_matrix.T.tocsr()

    if verbose:
        print("Reducing dimensions...")
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    vectors_reduced = svd.fit_transform(item_user_matrix)

    if verbose:
        print("Normalizing...")
    vectors_reduced /= (np.linalg.norm(vectors_reduced, axis=1, keepdims=True) + 1e-10)
    vectors_reduced = vectors_reduced.astype(np.float32)

    if verbose:
        print("Building Faiss index...")
    index = faiss.IndexFlatIP(n_components)
    index.add(vectors_reduced)

    if verbose:
        print("Searching for top-K similar items...")
    # D: cosine similarity scores, I: indices of top-K similar items
    query_vectors = vectors_reduced[transformed_ids]
    D, I = index.search(query_vectors, k)

    col_name = "article_id" if type == "item" else "customer_id"

    similar = pd.DataFrame({
        col_name: np.repeat(ids, k),
        'similarity': D.flatten(),
        "similar_object": [reverse_dict[ob] for ob in I.flatten()]
    })

    similar = similar[similar[col_name] != similar['similar_object']]
    similar = similar.reset_index(drop=True)

    return similar

if __name__ == "__main__":
    from src.models.generate_recommendations import load_data
    print("Loading data...")
    transactions, articles, customers, customer_map, reverse_customer_map = load_data()
    last_week = transactions['week'].max()
    train_window = 20
    print("Creating user-item matrix...")
    item_user_matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(transactions, last_week, train_window, matrix_type="uniform")
    similarity = top_k_cosine_similarity(item_user_matrix, [reverse_item_map[1], reverse_item_map[2], reverse_item_map[3]], item_map, reverse_item_map)
    print(similarity)
