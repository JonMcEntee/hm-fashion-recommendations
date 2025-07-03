import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.models.generate_recommendations import create_weekly_bestsellers_recommender

@pytest.fixture
def sample_transactions():
    """
    Create a sample transaction dataset for testing.
    """
    # Create sample data with known patterns
    dates = [
        datetime(2024, 1, 1),  # Week 0
        datetime(2024, 1, 1),  # Week 0
        datetime(2024, 1, 8),  # Week 1
        datetime(2024, 1, 8),  # Week 1
        datetime(2024, 1, 8),  # Week 1
        datetime(2024, 1, 15), # Week 2
    ]
    
    data = {
        't_dat': dates,
        'week': [0, 0, 1, 1, 1, 2],
        'customer_id': ['C1', 'C2', 'C1', 'C2', 'C3', 'C1'],
        'article_id': [1001, 1001, 1002, 1003, 1002, 1004]
    }
    return pd.DataFrame(data)

def test_bestsellers_basic_functionality(sample_transactions):
    """Test basic functionality of the weekly bestsellers recommender."""
    # Create recommender
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    # Get recommendations for week 2
    customers = ['C1', 'C2']
    recommendations = recommender(customers, week=2, k=2)
    
    # Check basic structure
    assert isinstance(recommendations, pd.DataFrame)
    assert set(recommendations.columns) == {'customer_id', 'article_id'}
    
    # Check we get correct number of recommendations
    assert len(recommendations) == 4  # 2 customers * 2 recommendations each
    
    # Check customer IDs are correct
    assert set(recommendations['customer_id'].unique()) == set(customers)
    

def test_bestsellers_week_isolation(sample_transactions):
    """Test that recommendations are based only on previous week's data."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    # Get recommendations for week 1
    # Should only consider items from week 0
    recommendations = recommender(['C1'], week=1, k=1)
    
    # In week 0, only 1001 was purchased
    assert recommendations['article_id'].iloc[0] == 1001

def test_bestsellers_ranking(sample_transactions):
    """Test that items are ranked by popularity."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    # Get recommendations for week 2
    # In week 1: 1002 appears twice, 1003 appears once
    recommendations = recommender(['C1'], week=2, k=2)
    
    # First recommendation should be 1002 (most popular in week 1)
    assert recommendations.iloc[0]['article_id'] == 1002
    # Second recommendation should be 1003
    assert recommendations.iloc[1]['article_id'] == 1003

def test_bestsellers_empty_week(sample_transactions):
    """Test behavior when requesting recommendations for a week with no prior data."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    # Get recommendations for week 0 (no prior week exists)
    recommendations = recommender(['C1'], week=0, k=1)
    
    # Should return empty recommendations
    assert len(recommendations) == 0  # Still returns a row for the customer

def test_bestsellers_k_parameter(sample_transactions):
    """Test that k parameter correctly limits number of recommendations."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    # Test with different k values
    for k in [1, 2, 3]:
        recommendations = recommender(['C1'], week=2, k=k)
        assert len(recommendations) <= k  # Should less than or equal to k recommendations

def test_bestsellers_multiple_customers(sample_transactions):
    """Test recommendations for multiple customers."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    customers = ['C1', 'C2', 'C3']
    k = 2
    recommendations = recommender(customers, week=2, k=k)
    
    # Check we get k recommendations for each customer
    assert len(recommendations) == len(customers) * k
    
    # Check each customer gets the same recommendations (bestsellers are global)
    c1_recs = recommendations[recommendations['customer_id'] == 'C1']['article_id'].tolist()
    for cust in customers[1:]:
        cust_recs = recommendations[recommendations['customer_id'] == cust]['article_id'].tolist()
        assert c1_recs == cust_recs

def test_bestsellers_data_types(sample_transactions):
    """Test that output DataFrame columns have correct data types for all weeks."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    customers = ['C1', 'C2']
    k = 2
    
    # Test for multiple weeks including week 0
    test_weeks = [1, 2]
    
    for week in test_weeks:
        recommendations = recommender(customers, week=week, k=k)    
        
        # Check data types
        assert recommendations['customer_id'].dtype == 'object', f"customer_id should be object type for week {week}"
        
        # Check recommendation column data type - should be integer for numeric article_ids
        if len(recommendations) > 0:
            assert recommendations['article_id'].dtype in ['int64', 'int32'], f"article_id should be integer type for week {week}"
        
        # Additional check: ensure no float values in recomme   ndation column
        if len(recommendations) > 0:
            recommendations_list = recommendations['article_id'].tolist()
            for rec in recommendations_list:
                # Check that recommendations are integers, not floats
                assert isinstance(rec, int), f"Recommendation {rec} should be integer, not float for week {week}"
                # Ensure no .0 suffix by checking the value equals its integer conversion
                assert rec == int(rec), f"Recommendation {rec} should be integer, not float for week {week}"

def test_bestsellers_max_k_per_customer(sample_transactions):
    """Test that each customer receives no more than k recommendations."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    customers = ['C1', 'C2', 'C3']
    for week in [1, 2]:
        for k in [1, 2, 3]:
            recommendations = recommender(customers, week=week, k=k)
            # Group by customer and count recommendations
            rec_counts = recommendations.groupby('customer_id').size()
            for customer in customers:
                assert rec_counts.get(customer, 0) <= k, f"Customer {customer} received more than {k} recommendations in week {week}" 