from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class CandidateGenerator(ABC):
    def __init__(self, transactions: pd.DataFrame, articles: pd.DataFrame):
        self.transactions = transactions
        self.articles = articles
        self.week = None
        self.context = None

    def set_context(self, context: dict):
        self.context = context

    @abstractmethod
    def set_week(self, week: int):
        pass

    def generate(self, users: List[str], k: int) -> pd.DataFrame:
        if self.week is None:
            raise ValueError("Week must be set before generating candidates")
        
        return self._generate(users, k)

    @abstractmethod
    def _generate(self, users: List[str], k: int) -> pd.DataFrame:
        pass