import pytest
import pandas as pd
from src.models.generate_recommendations import create_same_product_code

def test_basic_functionality():
    """Test that articles with the same product code are found correctly for each customer."""
    articles = pd.DataFrame({
        'article_id': [1001, 1002, 1003, 1004],
        'product_code': [100, 100, 200, 300]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C2'],
        'recommendation': [1001, 1003]
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # For C1 (1001, product_code 100): should get 1001, 1002
    # For C2 (1003, product_code 200): should get 1003
    c1_articles = set(result[result['customer_id'] == 'C1']['recommendation'])
    c2_articles = set(result[result['customer_id'] == 'C2']['recommendation'])
    assert c1_articles == {1001, 1002}
    assert c2_articles == {1003}

def test_multiple_previous_purchases_per_customer():
    """Test with multiple previous purchases for a single customer."""
    articles = pd.DataFrame({
        'article_id': [1001, 1002, 1003, 1004, 1005],
        'product_code': [100, 100, 200, 300, 200]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C1'],
        'recommendation': [1001, 1003]
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # C1 has 1001 (100) and 1003 (200), so should get 1001, 1002, 1003, 1005
    c1_articles = set(result[result['customer_id'] == 'C1']['recommendation'])
    assert c1_articles == {1001, 1002, 1003, 1005}

def test_no_matches():
    """Test when there are no matching product codes for a customer."""
    articles = pd.DataFrame({
        'article_id': [1001, 1002, 1003],
        'product_code': [100, 200, 300]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1'],
        'recommendation': [1004]  # 1004 not in articles
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # Should return empty DataFrame
    print(result)
    assert result.empty

def test_duplicates_removed():
    """Test that duplicate articles are not returned for a customer."""
    articles = pd.DataFrame({
        'article_id': [1001, 1002, 1003],
        'product_code': [100, 100, 100]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C1'],
        'recommendation': [1001, 1002]
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # Should return each article only once for C1
    c1_articles = result[result['customer_id'] == 'C1']['recommendation']
    assert set(c1_articles) == {1001, 1002, 1003}
    assert c1_articles.duplicated().sum() == 0

def test_empty_previous_purchases():
    """Test with empty previous purchases DataFrame."""
    articles = pd.DataFrame({
        'article_id': [1001, 1002],
        'product_code': [100, 200]
    })
    previous_purchases = pd.DataFrame({'customer_id': [], 'recommendation': []})
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # Should return empty DataFrame
    assert result.empty

def test_multiple_customers():
    """Test that each customer gets correct articles for their product codes."""
    articles = pd.DataFrame({
        'article_id': [1001, 1002, 1003, 1004, 1005],
        'product_code': [100, 100, 200, 300, 200]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C2'],
        'recommendation': [1001, 1003, 1004]
    })
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    # C1: 1001 (100) -> 1001, 1002
    # C2: 1003 (200), 1004 (300) -> 1003, 1005, 1004
    c1_articles = set(result[result['customer_id'] == 'C1']['recommendation'])
    c2_articles = set(result[result['customer_id'] == 'C2']['recommendation'])
    assert c1_articles == {1001, 1002}
    assert c2_articles == {1003, 1004, 1005}

def test_same_product_code_data_types():
    """Test that output DataFrame columns have correct data types."""
    articles = pd.DataFrame({
        'article_id': [1001, 1002, 1003, 1004, 1005],
        'product_code': [100, 100, 200, 300, 200]
    })
    previous_purchases = pd.DataFrame({
        'customer_id': ['C1', 'C2', 'C2'],
        'recommendation': [1001, 1003, 1004]
    })
    
    same_code_finder = create_same_product_code(articles)
    result = same_code_finder(previous_purchases)
    
    # Check that DataFrame is returned
    assert isinstance(result, pd.DataFrame)
    
    # Check column names
    expected_columns = {'customer_id', 'recommendation'}
    assert set(result.columns) == expected_columns
    
    # Check data types
    assert result['customer_id'].dtype == 'object', "customer_id should be object type"
    
    # Check recommendation column data type - should be integer for numeric article_ids
    if len(result) > 0:
        assert result['recommendation'].dtype in ['int64', 'int32'], "recommendation should be integer type"
    
    # Additional check: ensure no float values in recommendation column
    if len(result) > 0:
        recommendations_list = result['recommendation'].tolist()
        for rec in recommendations_list:
            # Check that recommendations are integers, not floats
            assert isinstance(rec, int), f"Recommendation {rec} should be integer, not float"
            # Ensure no .0 suffix by checking the value equals its integer conversion
            assert rec == int(rec), f"Recommendation {rec} should be integer, not float" 