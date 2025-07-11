from src.retrieval.candidate_generator import CandidateGenerator
import pandas as pd
from typing import List, Dict, Tuple, Any

class Ensemble(CandidateGenerator):
    def __init__(self,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        generators: Dict[str, Tuple[CandidateGenerator, Dict[str, Any]]]
    ):
        super().__init__(transactions, articles)
        self.generators = {}
        for name, (generator, kwargs) in generators.items():
            self.generators[name] = generator(transactions, articles, **kwargs)
            self.requirements.update(self.generators[name].requirements)

    def set_week(self, week: int) -> None:
        self.week = week
        for _, generator in self.generators.items():
            generator.set_week(week)

    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        recommendations = pd.DataFrame()
        self.context = {}

        for name, generator in self.generators.items():
            for requirement in generator.requirements:
                if requirement not in self.context:
                    raise ValueError((f"Requirement {requirement} for generator {name} "
                                      f"not found in context"))

            generator.set_context(self.context)
            new_recommendations = generator.generate(users, k)
            if name in self.requirements:
                self.context[name] = new_recommendations
            recommendations = pd.concat([recommendations, new_recommendations])

        recommendations = recommendations.sort_values(by="customer_id", ascending=False)

        return recommendations.drop_duplicates().reset_index(drop=True)


if __name__ == "__main__":
    from src.utils.data_load import load_data
    from src.retrieval.previous_purchases import PreviousPurchases
    from src.retrieval.same_product_code import SameProductCode
    from src.retrieval.item_similarity import ItemSimilarity

    print("Loading data...")
    transactions, articles, _, _, _ = load_data()
    customers = transactions["customer_id"].unique()[:5]

    print("Creating generators...")
    generators = {
        "previous_purchases": (PreviousPurchases, {"window_size": 25}),
        "same_product_code": (SameProductCode, {}),
        "item_similarity": (ItemSimilarity, {"train_window": 25, "matrix_type": "uniform", "n_components": 1000})
    }

    print("Creating ensemble...")
    ensemble = Ensemble(transactions, articles, generators)
    ensemble.set_week(50)

    print("Generating recommendations...")
    recommendations = ensemble.generate(customers, 3)

    print(recommendations)

