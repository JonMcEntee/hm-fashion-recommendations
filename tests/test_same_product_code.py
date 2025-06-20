import pytest
import pandas as pd
from src.models.generate_recommendations import create_same_product_code

def test_basic_functionality():
    """Test that articles with the same product code are found correctly for each customer."""
    articles = pd.DataFrame({
        'article_id': ['A1', 'A2', 'A3', 'A4'],
        'product_code': [100, 100, 200, 300]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C2'],
        'article_id': ['A1', 'A3']
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # For C1 (A1, product_code 100): should get A1, A2
    # For C2 (A3, product_code 200): should get A3
    c1_articles = set(result[result['customer_id'] == 'C1']['article_id'])
    c2_articles = set(result[result['customer_id'] == 'C2']['article_id'])
    assert c1_articles == {'A1', 'A2'}
    assert c2_articles == {'A3'}

def test_multiple_previous_purchases_per_customer():
    """Test with multiple previous purchases for a single customer."""
    articles = pd.DataFrame({
        'article_id': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'product_code': [100, 100, 200, 300, 200]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C1'],
        'article_id': ['A1', 'A3']
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # C1 has A1 (100) and A3 (200), so should get A1, A2, A3, A5
    c1_articles = set(result[result['customer_id'] == 'C1']['article_id'])
    assert c1_articles == {'A1', 'A2', 'A3', 'A5'}

def test_no_matches():
    """Test when there are no matching product codes for a customer."""
    articles = pd.DataFrame({
        'article_id': ['A1', 'A2', 'A3'],
        'product_code': [100, 200, 300]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1'],
        'article_id': ['A4']  # A4 not in articles
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # Should return empty DataFrame
    print(result)
    assert result.empty

def test_duplicates_removed():
    """Test that duplicate articles are not returned for a customer."""
    articles = pd.DataFrame({
        'article_id': ['A1', 'A2', 'A3'],
        'product_code': [100, 100, 100]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C1'],
        'article_id': ['A1', 'A2']
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # Should return each article only once for C1
    c1_articles = result[result['customer_id'] == 'C1']['article_id']
    assert set(c1_articles) == {'A1', 'A2', 'A3'}
    assert c1_articles.duplicated().sum() == 0

def test_empty_previous_purchases():
    """Test with empty previous purchases DataFrame."""
    articles = pd.DataFrame({
        'article_id': ['A1', 'A2'],
        'product_code': [100, 200]
    })
    previous_purchases = pd.DataFrame({'customer_id': [], 'article_id': []})
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # Should return empty DataFrame
    assert result.empty

def test_multiple_customers():
    """Test that each customer gets correct articles for their product codes."""
    articles = pd.DataFrame({
        'article_id': ['A1', 'A2', 'A3', 'A4', 'A5'],
        'product_code': [100, 100, 200, 300, 200]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C2'],
        'article_id': ['A1', 'A3', 'A4']
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # C1: A1 (100) -> A1, A2
    # C2: A3 (200), A4 (300) -> A3, A5, A4
    c1_articles = set(result[result['customer_id'] == 'C1']['article_id'])
    c2_articles = set(result[result['customer_id'] == 'C2']['article_id'])
    assert c1_articles == {'A1', 'A2'}
    assert c2_articles == {'A3', 'A4', 'A5'} 