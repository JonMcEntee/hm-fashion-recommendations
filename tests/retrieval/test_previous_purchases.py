import pytest
import pandas as pd
from datetime import datetime, timedelta
from retrieval.previous_purchases import PreviousPurchases

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
        'week': [0, 0, 1, 1, 1, 2],
        'customer_id': ['C1', 'C2', 'C1', 'C2', 'C1', 'C1'],
        'article_id': [1001, 1002, 1003, 1004, 1003, 1005]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_articles():
    """Create a sample articles dataset for testing."""
    data = {
        'article_id': [1001, 1002, 1003, 1004, 1005],
        'product_code': ['P1', 'P2', 'P3', 'P4', 'P5'],
        'product_type': ['Type1', 'Type2', 'Type3', 'Type4', 'Type5']
    }
    return pd.DataFrame(data)

def test_basic_functionality(sample_transactions, sample_articles):
    """Test basic functionality of the PreviousPurchases candidate generator."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    generator.set_week(1)
    
    # Get history for customer C1
    result = generator.generate(['C1'], k=10)
    
    # Check basic structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    
    # Check content
    # C1's history before week 1 should only include 1001 (from week 0)
    articles = set(result['article_id'])
    assert articles == {1001}
    
    # All rows should be for C1
    assert all(result['customer_id'] == 'C1')

def test_week_filtering(sample_transactions, sample_articles):
    """Test that only transactions before the specified week are included."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    
    # Test different weeks for the same customer
    generator.set_week(0)
    week0 = generator.generate(['C1'], k=10)
    
    generator.set_week(1)
    week1 = generator.generate(['C1'], k=10)
    
    generator.set_week(2)
    week2 = generator.generate(['C1'], k=10)
    
    # Week 0 should have no purchases (no data before week 0)
    assert set(week0['article_id']) == set()
    
    # Week 1 should only have 1001 (from week 0, before week 1)
    assert set(week1['article_id']) == {1001}
    
    # Week 2 should have 1001 and 1003 (from weeks 0 and 1, before week 2)
    assert set(week2['article_id']) == {1001, 1003}

def test_multiple_customers(sample_transactions, sample_articles):
    """Test retrieving history for multiple customers."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    generator.set_week(2)
    
    # Get history for both customers
    result = generator.generate(['C1', 'C2'], k=10)
    
    # Check we get histories for both customers
    assert set(result['customer_id']) == {'C1', 'C2'}
    
    # Check specific customer histories
    c1_articles = set(result[result['customer_id'] == 'C1']['article_id'])
    c2_articles = set(result[result['customer_id'] == 'C2']['article_id'])
    
    # C1: 1001 (week 0), 1003 (week 1) - both before week 2
    assert c1_articles == {1001, 1003}
    # C2: 1002 (week 0), 1004 (week 1) - both before week 2
    assert c2_articles == {1002, 1004}

def test_duplicate_purchases(sample_transactions, sample_articles):
    """Test that duplicate purchases are handled correctly (deduplicated)."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    generator.set_week(2)
    
    # C1 bought 1003 twice in week 1
    # Get history for week 2 (includes week 1 data)
    result = generator.generate(['C1'], k=10)
    
    # Should only appear once in the results
    a3_count = len(result[result['article_id'] == 1003])
    assert a3_count == 1

def test_nonexistent_customer(sample_transactions, sample_articles):
    """Test behavior with non-existent customer."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    generator.set_week(1)
    
    result = generator.generate(['NONEXISTENT'], k=10)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    assert len(result) == 0

def test_empty_customer_list(sample_transactions, sample_articles):
    """Test behavior with empty customer list."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    generator.set_week(1)
    
    result = generator.generate([], k=10)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    assert len(result) == 0


def test_data_types(sample_transactions, sample_articles):
    """Test that output DataFrame columns have correct data types for all weeks."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    
    customers = ['C1', 'C2']
    
    # Test for multiple weeks including week 0
    test_weeks = [0, 1, 2]
    
    for week in test_weeks:
        generator.set_week(week)
        result = generator.generate(customers, k=10)
        
        # Check that DataFrame is returned
        assert isinstance(result, pd.DataFrame)
        
        # Check column names
        expected_columns = {'customer_id', 'article_id'}
        assert set(result.columns) == expected_columns
        
        # Check data types
        assert result['customer_id'].dtype == 'object', f"customer_id should be object type for week {week}"
        
        # Check recommendation column data type - should be integer for numeric article_ids
        if len(result) > 0:
            assert result['article_id'].dtype in ['int64', 'int32'], f"article_id should be integer type for week {week}"
        
        # Additional check: ensure no float values in recommendation column
        if len(result) > 0:
            recommendations_list = result['article_id'].tolist()
            for rec in recommendations_list:
                # Check that recommendations are integers, not floats
                assert isinstance(rec, int), f"Recommendation {rec} should be integer, not float for week {week}"
                # Ensure no .0 suffix by checking the value equals its integer conversion
                assert rec == int(rec), f"Recommendation {rec} should be integer, not float for week {week}"
        
        # Verify expected content based on the "before week" behavior
        if week == 0:
            # Week 0 should have no purchases (no data before week 0)
            assert len(result) == 0
        elif week == 1:
            # Week 1 should only have purchases from week 0
            if len(result) > 0:
                expected_articles = {1001, 1002}  # From week 0
                actual_articles = set(result['article_id'])
                assert actual_articles.issubset(expected_articles), f"Week 1 should only contain articles from week 0: {actual_articles}"
        elif week == 2:
            # Week 2 should have purchases from weeks 0 and 1
            if len(result) > 0:
                expected_articles = {1001, 1002, 1003, 1004}  # From weeks 0 and 1
                actual_articles = set(result['article_id'])
                assert actual_articles.issubset(expected_articles), f"Week 2 should only contain articles from weeks 0 and 1: {actual_articles}"

def test_k_parameter(sample_transactions, sample_articles):
    """Test that the k parameter limits the number of recommendations per customer."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    generator.set_week(2)
    
    # C1 has 2 purchases before week 2: 1001 and 1003
    # Test with k=1 (should limit to 1 recommendation)
    result_k1 = generator.generate(['C1'], k=1)
    assert len(result_k1) == 1
    
    # Test with k=5 (should return all 2 recommendations)
    result_k5 = generator.generate(['C1'], k=5)
    assert len(result_k5) == 2

def test_window_size_parameter(sample_transactions, sample_articles):
    """Test that window_size parameter limits the historical window."""
    # Create transactions with more weeks
    extended_dates = [
        datetime(2024, 1, 1),   # Week 0
        datetime(2024, 1, 8),   # Week 1
        datetime(2024, 1, 15),  # Week 2
        datetime(2024, 1, 22),  # Week 3
        datetime(2024, 1, 29),  # Week 4
        datetime(2024, 2, 5),   # Week 5
    ]
    
    extended_data = {
        't_dat': extended_dates,
        'week': [0, 1, 2, 3, 4, 5],
        'customer_id': ['C1', 'C1', 'C1', 'C1', 'C1', 'C1'],
        'article_id': [1001, 1002, 1003, 1004, 1005, 1006]
    }
    extended_transactions = pd.DataFrame(extended_data)
    
    # Test with window_size=2 (should only include weeks 4 and 5 when requesting week 6)
    generator = PreviousPurchases(extended_transactions, sample_articles, window_size=2)
    generator.set_week(6)
    result = generator.generate(['C1'], k=10)
    
    # Should only include articles from weeks 4 and 5 (1005, 1006)
    assert set(result['article_id']) == {1005, 1006}

def test_generate_without_setting_week(sample_transactions, sample_articles):
    """Test that generating without setting week raises an error."""
    generator = PreviousPurchases(sample_transactions, sample_articles)
    
    with pytest.raises(ValueError, match="Week must be set before generating candidates"):
        generator.generate(['C1'], k=10)