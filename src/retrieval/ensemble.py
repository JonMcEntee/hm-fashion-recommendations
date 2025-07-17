"""
Ensemble Candidate Generator Module.

This module provides an ensemble candidate generator that combines multiple
individual candidate generators into a unified recommendation system. It implements
a sophisticated approach that manages dependencies between generators, shares
context between them, and aggregates their outputs into a comprehensive set of
recommendations.
"""

from retrieval.candidate_generator import CandidateGenerator
import pandas as pd
from typing import List, Dict, Tuple, Any

class Ensemble(CandidateGenerator):
    """
    An ensemble candidate generator that combines multiple individual generators.
    
    This generator orchestrates multiple candidate generators, managing their
    dependencies and sharing context between them. It supports complex recommendation
    pipelines where generators can depend on the outputs of other generators,
    enabling sophisticated multi-stage recommendation strategies.
    """
    
    def __init__(self,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        generators: Dict[str, Tuple[CandidateGenerator, Dict[str, Any]]]
    ):
        """
        Initialize the Ensemble candidate generator.
        
        Args:
            transactions (pd.DataFrame): DataFrame containing transaction data with
                                       columns: customer_id, article_id, week, t_dat
            articles (pd.DataFrame): DataFrame containing article metadata
            generators (Dict[str, Tuple[CandidateGenerator, Dict[str, Any]]]): 
                                       Dictionary mapping generator names to tuples of
                                       (generator_class, configuration_parameters)
        """
        super().__init__(transactions, articles)
        
        # Initialize storage for instantiated generators
        self.generators = {}
        
        # Instantiate each generator with its configuration parameters
        for name, (generator_class, kwargs) in generators.items():
            self.generators[name] = generator_class(transactions, articles, **kwargs)
            # Collect all requirements from child generators
            self.requirements.update(self.generators[name].requirements)

    def set_week(self, week: int) -> None:
        """
        Set the target week for all generators in the ensemble.
        
        This method propagates the target week to all child generators,
        ensuring they all operate on the same temporal context.
        
        Args:
            week (int): The target week for generating candidates
        """
        self.week = week
        
        # Set the week for all child generators
        for _, generator in self.generators.items():
            generator.set_week(week)

    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        """
        Generate candidate recommendations using the ensemble of generators.
        
        This method orchestrates the generation process by managing dependencies
        between generators, sharing context, and aggregating their outputs.
        Generators are executed in dependency order, with each generator's output
        potentially becoming input context for subsequent generators.
        
        Args:
            users (List[str]): List of user IDs to generate recommendations for
            k (int): Maximum number of recommendations per user
            
        Returns:
            pd.DataFrame: DataFrame with columns ['customer_id', 'article_id']
                         containing the aggregated recommendations from all generators
                         
        Raises:
            ValueError: If required dependencies are not found in context
        """
        # Initialize empty recommendations DataFrame and context dictionary
        recommendations = pd.DataFrame()
        self.context = {}

        # Process each generator in the ensemble
        for name, generator in self.generators.items():
            # Check if this generator's requirements are satisfied by the current context
            for requirement in generator.requirements:
                if requirement not in self.context:
                    raise ValueError((f"Requirement {requirement} for generator {name} "
                                      f"not found in context"))

            # Set the current context for this generator
            generator.set_context(self.context)
            
            # Generate recommendations using this generator
            new_recommendations = generator.generate(users, k)
            
            # If this generator's output is required by other generators, add it to context
            if name in self.requirements:
                self.context[name] = new_recommendations
            
            # Aggregate recommendations from this generator with previous ones
            recommendations = pd.concat([recommendations, new_recommendations])

        # Sort recommendations by customer_id for consistent ordering
        recommendations = recommendations.sort_values(by="customer_id", ascending=False)

        # Remove duplicates and reset index for clean output
        return recommendations.drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    """Example usage of the Ensemble candidate generator."""
    from utils.data_load import load_data
    from retrieval.previous_purchases import PreviousPurchases
    from retrieval.same_product_code import SameProductCode
    from retrieval.item_similarity import ItemSimilarity
    from retrieval.graph_search import GraphSearch

    print("Loading data...")
    transactions, articles, _, _, _ = load_data()
    customers = transactions["customer_id"].unique()[:5]

    print("Creating generators...")
    generators = {
        "previous_purchases": (PreviousPurchases, {"window_size": 25}),
        "same_product_code": (SameProductCode, {}),
        "item_similarity": (ItemSimilarity, {"train_window": 25, "matrix_type": "uniform", "n_components": 1000})
    }

    # generators = {
    #     "graph_search": (GraphSearch, {"window": 25, "max_steps": 10})
    # }

    print("Creating ensemble...")
    ensemble = Ensemble(transactions, articles, generators)
    ensemble.set_week(50)

    print("Generating recommendations...")
    recommendations = ensemble.generate(customers, 3)

    print("Generated recommendations:")
    print(recommendations)

