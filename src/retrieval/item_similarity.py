"""
Item Similarity Candidate Generator Module.

This module provides a candidate generator that recommends articles based on item-item
collaborative filtering using cosine similarity. It implements a sophisticated approach
that finds similar items based on user purchase patterns and recommends them to users
who have purchased related items.

The generator requires previous purchase recommendations as input.
"""

from retrieval.candidate_generator import CandidateGenerator
from retrieval.previous_purchases import PreviousPurchases
from models.collaborative_filtering import create_user_item_matrix, top_k_cosine_similarity
import pandas as pd
from typing import List

class ItemSimilarity(CandidateGenerator):
    """
    A candidate generator that recommends articles based on item-item collaborative filtering.
    
    This generator implements item-based collaborative filtering by computing cosine
    similarity between items based on user purchase patterns. It finds items that
    are frequently purchased together or by similar users, and recommends them to
    users who have purchased related items. The approach uses dimensionality reduction
    for computational efficiency.
    """
    
    def __init__(
            self,
            transactions: pd.DataFrame,
            articles: pd.DataFrame,
            train_window: int = 25,
            matrix_type: str = "uniform",
            n_components: int = 1000
        ):
        """
        Initialize the ItemSimilarity candidate generator.
        
        Args:
            transactions (pd.DataFrame): DataFrame containing transaction data with
                                       columns: customer_id, article_id, week, t_dat
            articles (pd.DataFrame): DataFrame containing article metadata
            train_window (int, optional): Number of weeks to look back for training
                                        data. Defaults to 25.
            matrix_type (str, optional): Type of user-item matrix to create.
                                       Options: "uniform", "weighted". Defaults to "uniform".
            n_components (int, optional): Number of components for dimensionality
                                        reduction in similarity computation. Defaults to 1000.
        """
        super().__init__(transactions, articles)
        
        # This generator requires previous purchase recommendations as input
        self.requirements = ["previous_purchases"]
        
        # Configuration parameters for collaborative filtering
        self.train_window = train_window
        self.matrix_type = matrix_type
        self.n_components = n_components
        
        # Storage for user-item matrix and mapping dictionaries
        self.item_user_matrix = None
        self.user_map = None
        self.item_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None

    def set_week(self, week: int) -> None:
        """
        Set the target week for candidate generation and build the user-item matrix.
        
        This method filters transactions up to the target week, creates the user-item
        matrix for collaborative filtering, and sets up the necessary mapping dictionaries
        for efficient similarity computation.
        
        Args:
            week (int): The target week for generating candidates
        """
        self.week = week
        
        # Filter transactions to only include data before the target week
        relevant_transactions = self.transactions[self.transactions["week"] < self.week]
        
        # Create user-item matrix and mapping dictionaries for collaborative filtering
        self.item_user_matrix, self.user_map, self.item_map, self.reverse_user_map, self.reverse_item_map = \
            create_user_item_matrix(relevant_transactions, self.week, self.train_window, self.matrix_type)
    
    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        """
        Generate candidate recommendations based on item similarity.
        
        This method takes previous purchase recommendations and finds similar items
        using cosine similarity computed from user purchase patterns. It uses
        collaborative filtering to discover items that are frequently purchased
        together or by similar users.
        
        Args:
            users (List[str]): List of user IDs to generate recommendations for
            k (int): Maximum number of recommendations per user
            
        Returns:
            pd.DataFrame: DataFrame with columns ['customer_id', 'article_id']
                         containing the recommended articles for each user
                         
        Raises:
            ValueError: If context is not set or previous_purchases not found in context
        """
        # Validate that context and required previous purchases are available
        if self.context is None:
            raise ValueError("Context must be set before generating candidates")
        
        if "previous_purchases" not in self.context:
            raise ValueError("Previous purchases must be set in context")
    
        # Get previous purchase recommendations for the specified users
        previous_purchases = self.context["previous_purchases"]
        previous_purchases = previous_purchases[previous_purchases["customer_id"].isin(users)]
        
        # Extract unique item IDs from previous purchases
        item_ids = previous_purchases['article_id'].unique()
        
        # Find similar items using cosine similarity with dimensionality reduction
        similar_items = top_k_cosine_similarity(
            self.item_user_matrix,
            item_ids,
            map_dict=self.item_map,
            reverse_dict=self.reverse_item_map,
            k=k,
            n_components=self.n_components
        )
        
        # Merge previous purchases with similar items and rank by similarity
        user_item_pairs = previous_purchases.merge(similar_items, on='article_id', how='left')\
            .sort_values(by=['customer_id', 'similarity'], ascending=False)\
            .groupby('customer_id', as_index=False)\
            .head(k)

        return user_item_pairs[['customer_id', 'article_id']]
    
if __name__ == "__main__":
    """Example usage of the ItemSimilarity candidate generator."""
    from utils.data_load import load_data
    
    print("Loading data...")
    transactions, articles, _, customer_map, reverse_customer_map = load_data()
    customers = transactions["customer_id"].unique()[:5]
    
    print("Creating previous purchases...")
    previous_purchases = PreviousPurchases(transactions, articles)
    previous_purchases.set_week(50)
    df = previous_purchases.generate(customers, k=10)
    
    print("Creating item similarity generator...")
    item_similarity = ItemSimilarity(transactions, articles)
    
    print("Setting context with previous purchases...")
    item_similarity.set_context({"previous_purchases": df})
    
    print("Setting target week...")
    item_similarity.set_week(50)
    
    print("Generating candidates...")
    result = item_similarity.generate(customers, k=10)
    
    print("Generated candidates:")
    print(result)