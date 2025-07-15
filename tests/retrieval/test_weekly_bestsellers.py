import pytest
import pandas as pd
from datetime import datetime, timedelta
from retrieval.weekly_bestsellers import WeeklyBestsellers

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

@pytest.fixture
def sample_articles():
    """
    Create a sample articles dataset for testing.
    """
    data = {
        'article_id': [1001, 1002, 1003, 1004],
        'product_code': ['P1', 'P2', 'P3', 'P4'],
        'prod_name': ['Product 1', 'Product 2', 'Product 3', 'Product 4'],
        'product_type': ['Type A', 'Type B', 'Type A', 'Type B'],
        'product_group': ['Group 1', 'Group 1', 'Group 2', 'Group 2']
    }
    return pd.DataFrame(data)

@pytest.fixture
def weekly_bestsellers(sample_transactions, sample_articles):
    """
    Create a WeeklyBestsellers instance for testing.
    """
    return WeeklyBestsellers(sample_transactions, sample_articles)

def test_initialization(sample_transactions, sample_articles):
    """Test that WeeklyBestsellers initializes correctly."""
    bestsellers = WeeklyBestsellers(sample_transactions, sample_articles)
    
    # Check that data is stored correctly
    assert bestsellers.transactions.equals(sample_transactions)
    assert bestsellers.articles.equals(sample_articles)
    assert bestsellers.week is None
    
    # Check that bestsellers_dict is created
    assert hasattr(bestsellers, 'best_sellers_dict')
    assert isinstance(bestsellers.best_sellers_dict, dict)

def test_bestsellers_dict_creation(sample_transactions, sample_articles):
    """Test that bestsellers_dict is created correctly."""
    bestsellers = WeeklyBestsellers(sample_transactions, sample_articles)
    
    # Week 0: only article 1001 was purchased
    # Week 1: article 1002 appears twice, 1003 appears once
    # Week 2: article 1004 appears once
    
    # Check week 1 (shifted from week 0)
    assert 1 in bestsellers.best_sellers_dict
    assert bestsellers.best_sellers_dict[1] == [1001]  # Only 1001 in week 0
    
    # Check week 2 (shifted from week 1)
    assert 2 in bestsellers.best_sellers_dict
    # 1002 should be first (appears twice), then 1003 (appears once)
    assert bestsellers.best_sellers_dict[2] == [1002, 1003]
    
    # Check week 3 (shifted from week 2)
    assert 3 in bestsellers.best_sellers_dict
    assert bestsellers.best_sellers_dict[3] == [1004]

def test_set_week_valid(weekly_bestsellers):
    """Test setting a valid week."""
    weekly_bestsellers.set_week(1)
    assert weekly_bestsellers.week == 1

def test_set_week_invalid(weekly_bestsellers):
    """Test setting an invalid week raises ValueError."""
    with pytest.raises(ValueError, match="Week 999 not found in data"):
        weekly_bestsellers.set_week(999)

def test_generate_without_setting_week(weekly_bestsellers):
    """Test that generate raises error if week is not set."""
    with pytest.raises(ValueError, match="Week must be set before generating candidates"):
        weekly_bestsellers.generate(['C1'], k=3)

def test_generate_basic_functionality(weekly_bestsellers):
    """Test basic functionality of generate method."""
    weekly_bestsellers.set_week(2)  # Use week 1 data (shifted)
    customers = ['C1', 'C2']
    recommendations = weekly_bestsellers.generate(customers, k=2)
    
    # Check basic structure
    assert isinstance(recommendations, pd.DataFrame)
    assert set(recommendations.columns) == {'customer_id', 'article_id'}
    
    # Check we get correct number of recommendations
    # Week 1 had 2 items: 1002 (rank 1), 1003 (rank 2)
    # 2 customers * 2 recommendations = 4 rows
    assert len(recommendations) == 4
    
    # Check customer IDs are correct
    assert set(recommendations['customer_id'].unique()) == set(customers)
    
    # Check article IDs are correct (should be 1002, 1003)
    expected_articles = [1002, 1003]
    actual_articles = recommendations['article_id'].unique().tolist()
    assert actual_articles == expected_articles

def test_generate_ranking_order(weekly_bestsellers):
    """Test that items are ranked by popularity."""
    weekly_bestsellers.set_week(2)  # Use week 1 data
    recommendations = weekly_bestsellers.generate(['C1'], k=2)
    
    # In week 1: 1002 appears twice, 1003 appears once
    # So 1002 should be first, then 1003
    assert recommendations.iloc[0]['article_id'] == 1002
    assert recommendations.iloc[1]['article_id'] == 1003

def test_generate_k_parameter(weekly_bestsellers):
    """Test that k parameter correctly limits number of recommendations."""
    weekly_bestsellers.set_week(2)  # Use week 1 data (has 2 items)
    
    # Test with k=1
    recommendations = weekly_bestsellers.generate(['C1'], k=1)
    assert len(recommendations) == 1
    assert recommendations.iloc[0]['article_id'] == 1002
    
    # Test with k=3 (more than available items)
    recommendations = weekly_bestsellers.generate(['C1'], k=3)
    assert len(recommendations) == 2  # Only 2 items available in week 1

def test_generate_multiple_customers(weekly_bestsellers):
    """Test recommendations for multiple customers."""
    weekly_bestsellers.set_week(2)  # Use week 1 data
    customers = ['C1', 'C2', 'C3']
    k = 2
    recommendations = weekly_bestsellers.generate(customers, k=k)
    
    # Check we get k recommendations for each customer
    assert len(recommendations) == len(customers) * k
    
    # Check each customer gets the same recommendations (bestsellers are global)
    c1_recs = recommendations[recommendations['customer_id'] == 'C1']['article_id'].tolist()
    for cust in customers[1:]:
        cust_recs = recommendations[recommendations['customer_id'] == cust]['article_id'].tolist()
        assert c1_recs == cust_recs

def test_generate_data_types(weekly_bestsellers):
    """Test that output DataFrame columns have correct data types."""
    weekly_bestsellers.set_week(2)
    customers = ['C1', 'C2']
    k = 2
    
    recommendations = weekly_bestsellers.generate(customers, k=k)
    
    # Check data types
    assert recommendations['customer_id'].dtype == 'object'
    assert recommendations['article_id'].dtype in ['int64', 'int32']
    
    # Check that article_id values are integers, not floats
    for article_id in recommendations['article_id']:
        assert isinstance(article_id, int)
        assert article_id == int(article_id)

def test_generate_max_k_per_customer(weekly_bestsellers):
    """Test that each customer receives no more than k recommendations."""
    weekly_bestsellers.set_week(2)
    customers = ['C1', 'C2', 'C3']
    
    for k in [1, 2, 3]:
        recommendations = weekly_bestsellers.generate(customers, k=k)
        # Group by customer and count recommendations
        rec_counts = recommendations.groupby('customer_id').size()
        for customer in customers:
            assert rec_counts.get(customer, 0) <= k, f"Customer {customer} received more than {k} recommendations"

def test_generate_customer_article_structure(weekly_bestsellers):
    """Test the structure of customer_id and article_id columns."""
    weekly_bestsellers.set_week(2)
    customers = ['C1', 'C2']
    k = 2
    
    recommendations = weekly_bestsellers.generate(customers, k=k)
    
    # Check that each customer appears exactly k times
    for customer in customers:
        customer_recs = recommendations[recommendations['customer_id'] == customer]
        assert len(customer_recs) == k
    
    # Check that article_id pattern is correct (repeated for each customer)
    expected_articles = [1002, 1003]  # From week 1 data
    for customer in customers:
        customer_recs = recommendations[recommendations['customer_id'] == customer]
        assert customer_recs['article_id'].tolist() == expected_articles

def test_inheritance_from_candidate_generator(weekly_bestsellers):
    """Test that WeeklyBestsellers properly inherits from CandidateGenerator."""
    from retrieval.candidate_generator import CandidateGenerator
    
    assert isinstance(weekly_bestsellers, CandidateGenerator)
    assert hasattr(weekly_bestsellers, 'set_week')
    assert hasattr(weekly_bestsellers, 'generate')
    assert hasattr(weekly_bestsellers, 'transactions')
    assert hasattr(weekly_bestsellers, 'articles')

def test_week_isolation(weekly_bestsellers):
    """Test that recommendations are based only on the specified week's data."""
    # Test week 2 (uses week 1 data)
    weekly_bestsellers.set_week(2)
    recommendations_week2 = weekly_bestsellers.generate(['C1'], k=2)
    
    # Test week 3 (uses week 2 data)
    weekly_bestsellers.set_week(3)
    recommendations_week3 = weekly_bestsellers.generate(['C1'], k=2)
    
    # Should be different recommendations
    assert not recommendations_week2.equals(recommendations_week3)
    
    # Week 2 recommendations should be from week 1 data: [1002, 1003]
    assert recommendations_week2['article_id'].tolist() == [1002, 1003]
    
    # Week 3 recommendations should be from week 2 data: [1004]
    assert recommendations_week3['article_id'].tolist() == [1004] 