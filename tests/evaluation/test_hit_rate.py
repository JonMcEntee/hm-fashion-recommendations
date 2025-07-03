import pandas as pd
import pytest
from src.evaluation.metrics import hit_rate

def test_hit_rate_basic():
    recommendations = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C1', 'C3'],
        'article_id': ['A1', 'A2', 'A3', 'A4'],
        'week': [1, 1, 2, 2]
    })
    transactions = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C1', 'C3', 'C2'],
        'article_id': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'week': [1, 1, 2, 2, 2]
    })
    result = hit_rate(recommendations, transactions)
    # Week 1: 2 transactions, both covered
    # Week 2: 3 transactions, 2 covered
    row1 = result[result['week'] == 1].iloc[0]
    row2 = result[result['week'] == 2].iloc[0]
    assert row1['covered'] == 2
    assert row1['total'] == 2
    assert row1['percent'] == 1.0
    assert row2['covered'] == 2
    assert row2['total'] == 3
    assert row2['percent'] == 2/3

def test_hit_rate_no_coverage():
    recommendations = pd.DataFrame({
        'customer_id': ['C1'],
        'article_id': ['A9'],
        'week': [1]
    })
    transactions = pd.DataFrame({
        'customer_id': ['C1', 'C2'],
        'article_id': ['A1', 'A2'],
        'week': [1, 1]
    })
    result = hit_rate(recommendations, transactions)
    assert (result['covered'] == 0).all()
    assert (result['percent'] == 0).all()

def test_hit_rate_empty():
    recommendations = pd.DataFrame(columns=['customer_id', 'article_id', 'week'])
    transactions = pd.DataFrame(columns=['customer_id', 'article_id', 'week'])
    result = hit_rate(recommendations, transactions)
    assert result.empty

def test_hit_rate_partial_coverage():
    recommendations = pd.DataFrame({
        'customer_id': ['C1', 'C2'],
        'article_id': ['A1', 'A2'],
        'week': [1, 1]
    })
    transactions = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C3'],
        'article_id': ['A1', 'A2', 'A3'],
        'week': [1, 1, 1]
    })
    result = hit_rate(recommendations, transactions)
    row = result[result['week'] == 1].iloc[0]
    assert row['covered'] == 2
    assert row['total'] == 3
    assert row['percent'] == 2/3 