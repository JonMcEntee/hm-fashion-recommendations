"""
Same Product Code Candidate Generator Module.

This module provides a candidate generator that recommends articles based on users'
previous purchases by suggesting other articles with the same product code. It
implements an approach that leverages product similarity to expand user preferences
beyond their exact purchase history.

The generator requires previous purchase recommendations as input and uses time-decay
weighting to prioritize more recent product popularity when ranking candidates.
"""

from retrieval.candidate_generator import CandidateGenerator
import pandas as pd
from typing import List

class SameProductCode(CandidateGenerator):
    """
    A candidate generator that recommends articles with the same product code as previous purchases.
    
    This generator suggests articles that share the same product code as items the 
    user has purchased before. It uses time-decay weighting to prioritize products
    that have been popular  more recently, and requires previous purchase recommendations
    as input context.
    """
    
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame):
        """
        Initialize the SameProductCode candidate generator.
        
        Args:
            transactions (pd.DataFrame): DataFrame containing transaction data with
                                       columns: customer_id, article_id, week, t_dat
            articles (pd.DataFrame): DataFrame containing article metadata with
                                   columns: article_id, product_code
        """
        super().__init__(transactions, articles)
        
        # This generator requires previous purchase recommendations as input
        self.requirements = ["previous_purchases"]

        # Prepare transaction data for time-decay weighting
        transactions = transactions[['customer_id', 'week', 'article_id', 't_dat']].copy()
    
        # Calculate time-decay weights: more recent weeks get higher weights
        last_week = transactions['week'].max()
        transactions["time_decay"] = transactions["week"].apply(lambda x: 0.95 ** (last_week - x))
        
        # Aggregate time-decay weights by article and week, then calculate cumulative weights
        weighted_transactions = transactions.groupby(['article_id', 'week'])['time_decay'].agg('sum').reset_index()
        weighted_transactions['cumulative_weight'] = weighted_transactions\
            .sort_values(by='week', ascending=True)\
            .groupby(['article_id'])["time_decay"]\
            .cumsum()
        
        self.weighted_transactions = weighted_transactions

        # Build mapping from article_id to list of similar articles (same product_code)
        index = articles[['article_id', 'product_code']].copy()

        # Group by product_code to get all article_ids for each product_code
        product_code_to_articles = index.groupby('product_code')['article_id'].apply(list)

        # Map each article_id to its list of similar articles (same product_code)
        index['similar_articles'] = index['product_code'].map(product_code_to_articles)

        # Convert to dict: article_id -> list of similar articles for fast lookup
        self.similar_articles = dict(zip(index['article_id'], index['similar_articles']))

    def set_week(self, week: int) -> None:
        """
        Set the target week for candidate generation.
        
        This method stores the target week for use in filtering relevant
        weighted transactions during candidate generation.
        
        Args:
            week (int): The target week for generating candidates
        """
        self.week = week
        
    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        """
        Generate candidate recommendations based on same product code.
        
        This method takes previous purchase recommendations and expands them to
        include articles with the same product code. It uses time-decay weighted
        popularity to rank the expanded candidates for each user.
        
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
                
        # Start with previous purchase recommendations and expand to same product codes
        same_code = self.context["previous_purchases"][['customer_id', 'article_id']].copy()        
        same_code = same_code[same_code["customer_id"].isin(users)]
        
        # Map each purchased article to all articles with the same product code
        same_code["article_id"] = same_code["article_id"].map(self.similar_articles)
        
        # Explode the list of similar articles into separate rows
        same_code = same_code.explode("article_id").astype({"article_id": "int32"})

        # Filter weighted transactions to only include data before the target week
        relevant_weights = self.weighted_transactions[self.weighted_transactions["week"] < self.week].copy()

        # Add a column to track the maximum week for each article_id
        relevant_weights["max_week"] = relevant_weights.groupby("article_id")["week"].transform("max")

        # Keep only the most recent week's data for each article_id
        relevant_weights = relevant_weights[relevant_weights["max_week"] == relevant_weights["week"]]

        # Clean up temporary columns
        relevant_weights = relevant_weights.drop(columns=["max_week", "week"])

        # Merge candidates with their cumulative weights, sort by weight, and take top k per user
        same_code = (
            same_code
            .drop_duplicates()  # Remove duplicate user-article pairs
            .merge(relevant_weights, on=['article_id'], how='left')  # Add popularity weights
            .sort_values(by=['customer_id', 'cumulative_weight'], ascending=False)  # Sort by popularity
            .groupby('customer_id', as_index=False)  # Group by user
            .head(k)  # Take top k recommendations per user
            .reset_index(drop=True)
        )

        return same_code[['customer_id', 'article_id']]

if __name__ == "__main__":
    """Example usage of the SameProductCode candidate generator."""
    from utils.data_load import load_data
    from retrieval.previous_purchases import PreviousPurchases
    
    print("Loading data...")
    transactions, articles, customers, customer_map, reverse_customer_map = load_data()
    customers = transactions["customer_id"].unique()[:5]
    
    print("Creating previous purchases...")
    previous_purchases = PreviousPurchases(transactions, articles)
    previous_purchases.set_week(50)
    result = previous_purchases.generate(customers, k=10)
    
    print("Creating same product code generator...")
    same_product_code = SameProductCode(transactions, articles)
    
    print("Setting target week...")
    same_product_code.set_week(50)
    
    print("Setting context with previous purchases...")
    same_product_code.set_context({"previous_purchases": result})
    
    print("Generating candidates...")
    candidates = same_product_code.generate(customers, k=3)
    
    print("Generated candidates:")
    print(candidates)
    print("Recommendations per customer:")
    print(candidates["customer_id"].value_counts())