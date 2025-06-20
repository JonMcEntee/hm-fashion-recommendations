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
        'customer_id': ['C1', 'C2', 'C1', 'C2', 'C3', 'C1'],
        'article_id': ['A1', 'A1', 'A2', 'A3', 'A2', 'A4']
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
    assert set(recommendations.columns) == {'customer_id', 'rank', 'recommendation'}
    
    # Check we get correct number of recommendations
    assert len(recommendations) == 4  # 2 customers * 2 recommendations each
    
    # Check customer IDs are correct
    assert set(recommendations['customer_id'].unique()) == set(customers)
    
    # Check ranks are correct
    assert list(recommendations['rank'].unique()) == [1, 2]

def test_bestsellers_week_isolation(sample_transactions):
    """Test that recommendations are based only on previous week's data."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    # Get recommendations for week 1
    # Should only consider items from week 0
    recommendations = recommender(['C1'], week=1, k=1)
    
    # In week 0, only A1 was purchased
    assert recommendations['recommendation'].iloc[0] == 'A1'

def test_bestsellers_ranking(sample_transactions):
    """Test that items are ranked by popularity."""
    recommender = create_weekly_bestsellers_recommender(sample_transactions)
    
    # Get recommendations for week 2
    # In week 1: A2 appears twice, A3 appears once
    recommendations = recommender(['C1'], week=2, k=2)
    
    # First recommendation should be A2 (most popular in week 1)
    assert recommendations.iloc[0]['recommendation'] == 'A2'
    # Second recommendation should be A3
    assert recommendations.iloc[1]['recommendation'] == 'A3'

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
    c1_recs = recommendations[recommendations['customer_id'] == 'C1']['recommendation'].tolist()
    for cust in customers[1:]:
        cust_recs = recommendations[recommendations['customer_id'] == cust]['recommendation'].tolist()
        assert c1_recs == cust_recs 