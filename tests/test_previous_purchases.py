import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.models.generate_recommendations import create_previous_purchases

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
        'article_id': [1001, 1002, 1003, 1004, 1003, 1005]
    }
    return pd.DataFrame(data)

def test_basic_functionality(sample_transactions):
    """Test basic functionality of the customer item recommender."""
    recommender = create_previous_purchases(sample_transactions)
    
    # Get history for customer C1 at week 1
    result = recommender(['C1'], week=1)
    
    # Check basic structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'recommendation'}
    
    # Check content
    # C1's history up to week 1 should include 1001 and 1003
    articles = set(result['recommendation'])
    assert articles == {1001, 1003}
    
    # All rows should be for C1
    assert all(result['customer_id'] == 'C1')

def test_week_filtering(sample_transactions):
    """Test that only transactions up to the specified week are included."""
    recommender = create_previous_purchases(sample_transactions)
    
    # Test different weeks for the same customer
    week0 = recommender(['C1'], week=0)
    week1 = recommender(['C1'], week=1)
    week2 = recommender(['C1'], week=2)
    
    # Week 0 should only have 1001
    assert set(week0['recommendation']) == {1001}
    
    # Week 1 should have 1001 and 1003
    assert set(week1['recommendation']) == {1001, 1003}
    
    # Week 2 should have 1001, 1003, and 1005
    assert set(week2['recommendation']) == {1001, 1003, 1005}

def test_multiple_customers(sample_transactions):
    """Test retrieving history for multiple customers."""
    recommender = create_previous_purchases(sample_transactions)
    
    # Get history for both customers at week 2
    result = recommender(['C1', 'C2'], week=2)
    
    # Check we get histories for both customers
    assert set(result['customer_id']) == {'C1', 'C2'}
    
    # Check specific customer histories
    c1_articles = set(result[result['customer_id'] == 'C1']['recommendation'])
    c2_articles = set(result[result['customer_id'] == 'C2']['recommendation'])
    
    assert c1_articles == {1001, 1003, 1005}
    assert c2_articles == {1002, 1004}

def test_duplicate_purchases(sample_transactions):
    """Test that duplicate purchases are handled correctly (deduplicated)."""
    recommender = create_previous_purchases(sample_transactions)
    
    # C1 bought 1003 twice in week 1
    result = recommender(['C1'], week=1)
    
    # Should only appear once in the results
    a3_count = len(result[result['recommendation'] == 1003])
    assert a3_count == 1

def test_nonexistent_customer(sample_transactions):
    """Test behavior with non-existent customer."""
    recommender = create_previous_purchases(sample_transactions)
    
    result = recommender(['NONEXISTENT'], week=1)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'recommendation'}
    assert len(result) == 0

def test_empty_customer_list(sample_transactions):
    """Test behavior with empty customer list."""
    recommender = create_previous_purchases(sample_transactions)
    
    result = recommender([], week=1)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'recommendation'}
    assert len(result) == 0

def test_future_week(sample_transactions):
    """Test behavior when requesting a future week."""
    recommender = create_previous_purchases(sample_transactions)
    
    # Request week beyond the data
    result = recommender(['C1'], week=10)
    
    # Should return all historical purchases for the customer
    assert set(result['recommendation']) == {1001, 1003, 1005}

def test_negative_week(sample_transactions):
    """Test behavior with negative week number."""
    recommender = create_previous_purchases(sample_transactions)
    
    result = recommender(['C1'], week=-1)
    
    # Should return empty history as no transactions exist before week 0
    assert len(result) == 0

def test_previous_purchases_data_types(sample_transactions):
    """Test that output DataFrame columns have correct data types for all weeks."""
    recommender = create_previous_purchases(sample_transactions)
    
    customers = ['C1', 'C2']
    
    # Test for multiple weeks including week 0
    test_weeks = [0, 1, 2]
    
    for week in test_weeks:
        result = recommender(customers, week=week)
        
        # Check that DataFrame is returned
        assert isinstance(result, pd.DataFrame)
        
        # Check column names
        expected_columns = {'customer_id', 'recommendation'}
        assert set(result.columns) == expected_columns
        
        # Check data types
        assert result['customer_id'].dtype == 'object', f"customer_id should be object type for week {week}"
        
        # Check recommendation column data type - should be integer for numeric article_ids
        if len(result) > 0:
            assert result['recommendation'].dtype in ['int64', 'int32'], f"recommendation should be integer type for week {week}"
        
        # Additional check: ensure no float values in recommendation column
        if len(result) > 0:
            recommendations_list = result['recommendation'].tolist()
            for rec in recommendations_list:
                # Check that recommendations are integers, not floats
                assert isinstance(rec, int), f"Recommendation {rec} should be integer, not float for week {week}"
                # Ensure no .0 suffix by checking the value equals its integer conversion
                assert rec == int(rec), f"Recommendation {rec} should be integer, not float for week {week}"