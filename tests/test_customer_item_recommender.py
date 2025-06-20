import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.models.generate_recommendations import create_customer_item_recommender

@pytest.fixture
def sample_transactions():
    """
    Create a sample transaction dataset for testing.
    Contains transactions across multiple weeks with known patterns.
    """
    dates = [
        datetime(2024, 1, 1),   # Week 0
        datetime(2024, 1, 1),   # Week 0
        datetime(2024, 1, 8),   # Week 1
        datetime(2024, 1, 8),   # Week 1
        datetime(2024, 1, 8),   # Week 1 (duplicate purchase)
        datetime(2024, 1, 15),  # Week 2
    ]
    
    data = {
        't_dat': dates,
        'customer_id': ['C1', 'C2', 'C1', 'C2', 'C1', 'C1'],
        'article_id': ['A1', 'A2', 'A3', 'A4', 'A3', 'A5']
    }
    return pd.DataFrame(data)

def test_basic_functionality(sample_transactions):
    """Test basic functionality of the customer item recommender."""
    recommender = create_customer_item_recommender(sample_transactions)
    
    # Get history for customer C1 at week 1
    result = recommender(['C1'], week=1)
    
    # Check basic structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    
    # Check content
    # C1's history up to week 1 should include A1 and A3
    articles = set(result['article_id'])
    assert articles == {'A1', 'A3'}
    
    # All rows should be for C1
    assert all(result['customer_id'] == 'C1')

def test_week_filtering(sample_transactions):
    """Test that only transactions up to the specified week are included."""
    recommender = create_customer_item_recommender(sample_transactions)
    
    # Test different weeks for the same customer
    week0 = recommender(['C1'], week=0)
    week1 = recommender(['C1'], week=1)
    week2 = recommender(['C1'], week=2)
    
    # Week 0 should only have A1
    assert set(week0['article_id']) == {'A1'}
    
    # Week 1 should have A1 and A3
    assert set(week1['article_id']) == {'A1', 'A3'}
    
    # Week 2 should have A1, A3, and A5
    assert set(week2['article_id']) == {'A1', 'A3', 'A5'}

def test_multiple_customers(sample_transactions):
    """Test retrieving history for multiple customers."""
    recommender = create_customer_item_recommender(sample_transactions)
    
    # Get history for both customers at week 2
    result = recommender(['C1', 'C2'], week=2)
    
    # Check we get histories for both customers
    assert set(result['customer_id']) == {'C1', 'C2'}
    
    # Check specific customer histories
    c1_articles = set(result[result['customer_id'] == 'C1']['article_id'])
    c2_articles = set(result[result['customer_id'] == 'C2']['article_id'])
    
    assert c1_articles == {'A1', 'A3', 'A5'}
    assert c2_articles == {'A2', 'A4'}

def test_duplicate_purchases(sample_transactions):
    """Test that duplicate purchases are handled correctly (deduplicated)."""
    recommender = create_customer_item_recommender(sample_transactions)
    
    # C1 bought A3 twice in week 1
    result = recommender(['C1'], week=1)
    
    # Should only appear once in the results
    a3_count = len(result[result['article_id'] == 'A3'])
    assert a3_count == 1

def test_nonexistent_customer(sample_transactions):
    """Test behavior with non-existent customer."""
    recommender = create_customer_item_recommender(sample_transactions)
    
    result = recommender(['NONEXISTENT'], week=1)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    assert len(result) == 0

def test_empty_customer_list(sample_transactions):
    """Test behavior with empty customer list."""
    recommender = create_customer_item_recommender(sample_transactions)
    
    result = recommender([], week=1)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    assert len(result) == 0

def test_future_week(sample_transactions):
    """Test behavior when requesting a future week."""
    recommender = create_customer_item_recommender(sample_transactions)
    
    # Request week beyond the data
    result = recommender(['C1'], week=10)
    
    # Should return all historical purchases for the customer
    assert set(result['article_id']) == {'A1', 'A3', 'A5'}

def test_negative_week(sample_transactions):
    """Test behavior with negative week number."""
    recommender = create_customer_item_recommender(sample_transactions)
    
    result = recommender(['C1'], week=-1)
    
    # Should return empty history as no transactions exist before week 0
    assert len(result) == 0