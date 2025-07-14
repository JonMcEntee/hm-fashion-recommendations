import pandas as pd
import os
from typing import List, Callable, Dict, Tuple, Any
from src.retrieval.candidate_generator import CandidateGenerator
from tqdm import tqdm

class BatchGenerator:
    def __init__(self,
        transactions: pd.DataFrame,
        articles: pd.DataFrame,
        generators: Dict[str, Tuple[CandidateGenerator, Dict[str, Any], bool]],
        k: int = 100,
        first_week: int = 50,
        verbose: bool = False,
        to_csv: bool = False,
        overwrite: bool = False,
        path: str = "data/recommendations.csv",
        metrics_path: str = "data/metrics.csv"
    ):
        self.hits = transactions[['customer_id', 'article_id', 'week']].drop_duplicates()
        self.path = path
        self.metrics_path = metrics_path
        self.overwrite = overwrite
        self.k = k
        self.first_week = first_week
        self.last_week = self.hits['week'].max()
        self.verbose = verbose
        self.to_csv = to_csv
        self.generators = {}
        self.requirements = set()
        self.retain = {}

        for name, (generator_class, kwargs, retain) in generators.items():
            self.generators[name] = generator_class(transactions, articles, **kwargs)
            self.retain[name] = retain
            self.requirements.update(self.generators[name].requirements)

    def set_week(self, week: int) -> None:
        self.week = week
        
        for _, generator in self.generators.items():
            generator.set_week(week)

    def generate_batch(self, metrics: bool = False) -> pd.DataFrame:
        coverage = pd.DataFrame(columns=["generator", "hit_rate", "week"])
        recommendations = pd.DataFrame(columns=["customer_id", "article_id", "week"])

        if self.to_csv:
            if self.overwrite:
                os.remove(self.metrics_path)
                os.remove(self.path)
            elif os.path.exists(self.path):
                raise FileExistsError(f"File {self.path} already exists. Set overwrite=True to overwrite.")
            elif os.path.exists(self.metrics_path):
                raise FileExistsError(f"File {self.metrics_path} already exists. Set overwrite=True to overwrite.")

            coverage.to_csv(self.metrics_path, index=False)
            recommendations.to_csv(self.path, index=False)

        for week in tqdm(range(self.first_week, self.last_week + 1), desc="Generating recommendations", disable=not self.verbose):
            users = self.hits[self.hits['week'] == week]['customer_id'].unique()
            hits = self.hits[self.hits['week'] == week][["customer_id", "article_id"]].drop_duplicates()
            recommendations_, coverage_ = self._generate_week(users, hits, week, metrics)

            recommendations_['week'] = week
            coverage_['week'] = week

            if self.to_csv:
                recommendations_.to_csv(self.path.format(week), mode='a', index=False, header=False)
            else:
                recommendations = pd.concat([recommendations, recommendations_])

            if metrics:
                if self.to_csv:
                    coverage_.to_csv(self.metrics_path, mode='a', index=False, header=False)
                else:
                    coverage = pd.concat([coverage, coverage_])

        return recommendations, coverage

    def _generate_week(self, users: List[str], hits: pd.DataFrame, week: int, metrics: bool = False) -> pd.DataFrame:
        self.context = {}
        self.set_week(week)
        rec_index = pd.MultiIndex.from_frame(hits)

        recommendation_batch = pd.DataFrame()
        coverage = [] if metrics else None

        for name, generator in self.generators.items():
            for requirement in generator.requirements:
                if requirement not in self.context:
                    raise ValueError((f"Requirement {requirement} for generator {name} "
                                      f"not found in context"))

            generator.set_context(self.context)
            recommendations = generator.generate(users, self.k)

            if name in self.requirements:
                self.context[name] = recommendations

            if self.retain[name]:
                recommendation_batch = pd.concat([recommendation_batch, recommendations])
                if metrics:
                    trans_index = pd.MultiIndex.from_frame(recommendations)
                    covered = trans_index.isin(rec_index)
                    coverage.append({"generator": name, "hit_rate": covered.sum() / len(rec_index)})
        
        recommendation_batch = recommendation_batch.drop_duplicates().reset_index(drop=True)
        coverage = pd.DataFrame(coverage)

        return recommendation_batch, coverage

if __name__ == "__main__":
    from src.retrieval.previous_purchases import PreviousPurchases
    from src.retrieval.same_product_code import SameProductCode
    from src.retrieval.item_similarity import ItemSimilarity
    from src.utils.data_load import load_data

    print("Loading data...")
    transactions, articles, customers, customer_map, reverse_customer_map = load_data()

    generators = {
        "previous_purchases": (PreviousPurchases, {"window_size": 25}, True),
        "same_product_code": (SameProductCode, {}, True),
        "item_similarity": (ItemSimilarity, {"train_window": 25, "matrix_type": "uniform", "n_components": 1000}, True)
    }

    print("Loading generators...")
    batch_generator = BatchGenerator(
        transactions,
        articles,
        generators,
        to_csv=True,
        verbose=True,
        overwrite=True,
        path="data/recommendations.csv",
        metrics_path="results/metrics.csv"
    )

    batch_generator.generate_batch(metrics=True)