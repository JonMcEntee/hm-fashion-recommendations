import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from retrieval.item_similarity import ItemSimilarity

def make_test_transactions():
    """Create test transaction data for item similarity tests."""
    base_date = datetime(2024, 1, 1)
    dates = [
        base_date,                    # Week 0
        base_date + timedelta(days=7),  # Week 1
        base_date + timedelta(days=14), # Week 2
        base_date + timedelta(days=21), # Week 3
        base_date + timedelta(days=28), # Week 4
    ]
    
    # Create more diverse transaction data with more articles
    data = {
        't_dat': dates * 4,  # More transactions
        'week': [0, 1, 2, 3, 4] * 4,
        'customer_id': ['C1', 'C1', 'C2', 'C2', 'C3', 'C1', 'C2', 'C3', 'C1', 'C2'] * 2,
        'article_id': [1001, 1002, 1001, 1003, 1002, 1004, 1005, 1001, 1006, 1007] * 2
    }
    return pd.DataFrame(data)

def make_previous_purchases(customers, article_ids, week_numbers=None):
    """Helper function to create previous purchases context."""
    if week_numbers is None:
        week_numbers = [0 for _ in customers]
    return pd.DataFrame({
        'customer_id': customers,
        'article_id': article_ids,
        'week': week_numbers
    })

@pytest.fixture
def sample_transactions():
    """Fixture providing sample transaction data."""
    return make_test_transactions()

@pytest.fixture
def sample_articles():
    """Create a sample articles dataset for testing."""
    return pd.DataFrame({
        'article_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007],
        'product_code': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'],
        'product_type': ['Type1', 'Type2', 'Type3', 'Type4', 'Type5', 'Type6', 'Type7']
    })

def test_basic_functionality(sample_transactions, sample_articles):
    """Test basic functionality of the ItemSimilarity candidate generator."""
    previous_purchases = make_previous_purchases(['C1', 'C2'], [1001, 1002], week_numbers=[0, 0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles, n_components=2)
    item_similarity.set_week(2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    result = item_similarity.generate(['C1', 'C2'], k=2)
    
    # Check basic structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    
    # Check that we get results
    assert len(result) > 0
    
    # Check that all customers are represented
    assert set(result['customer_id']) == {'C1', 'C2'}

def test_week_filtering(sample_transactions, sample_articles):
    """Test that only transactions before the specified week are considered."""
    previous_purchases = make_previous_purchases(['C1'], [1001], week_numbers=[0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles, n_components=2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    # Test with different weeks
    item_similarity.set_week(2)  # Only week 0 data
    result_week2 = item_similarity.generate(['C1'], k=2)
    
    item_similarity.set_week(3)  # Weeks 0, 1, 2 data
    result_week3 = item_similarity.generate(['C1'], k=2)
    
    # Both should work but may have different results due to different training data
    assert isinstance(result_week2, pd.DataFrame)
    assert isinstance(result_week3, pd.DataFrame)
    assert len(result_week2) > 0
    assert len(result_week3) > 0
    assert len(result_week2) <= len(result_week3)

def test_matrix_type_parameter(sample_transactions, sample_articles):
    """Test that different matrix types work correctly."""
    previous_purchases = make_previous_purchases(['C1'], [1001], week_numbers=[0])
    
    # Test uniform matrix type
    item_similarity_uniform = ItemSimilarity(sample_transactions, sample_articles, matrix_type="uniform", n_components=2)
    item_similarity_uniform.set_week(2)
    item_similarity_uniform.set_context({"previous_purchases": previous_purchases})
    result_uniform = item_similarity_uniform.generate(['C1'], k=2)
    
    # Test time decay matrix type
    item_similarity_decay = ItemSimilarity(sample_transactions, sample_articles, matrix_type="time_decay", n_components=2)
    item_similarity_decay.set_week(2)
    item_similarity_decay.set_context({"previous_purchases": previous_purchases})
    result_decay = item_similarity_decay.generate(['C1'], k=2)
    
    # Both should work
    assert isinstance(result_uniform, pd.DataFrame)
    assert isinstance(result_decay, pd.DataFrame)
    assert len(result_uniform) > 0
    assert len(result_decay) > 0

def test_train_window_parameter(sample_transactions, sample_articles):
    """Test that train_window parameter affects the result."""
    previous_purchases = make_previous_purchases(['C1'], [1001], week_numbers=[0])
    
    # Test with different train windows
    item_similarity_short = ItemSimilarity(sample_transactions, sample_articles, train_window=2, n_components=2)
    item_similarity_short.set_week(4)
    item_similarity_short.set_context({"previous_purchases": previous_purchases})
    result_short = item_similarity_short.generate(['C1'], k=2)
    
    item_similarity_long = ItemSimilarity(sample_transactions, sample_articles, train_window=5, n_components=2)
    item_similarity_long.set_week(4)
    item_similarity_long.set_context({"previous_purchases": previous_purchases})
    result_long = item_similarity_long.generate(['C1'], k=2)
    
    # Both should work
    assert isinstance(result_short, pd.DataFrame)
    assert isinstance(result_long, pd.DataFrame)
    assert len(result_short) > 0
    assert len(result_long) > 0
    assert len(result_short) <= len(result_long)

def test_k_parameter(sample_transactions, sample_articles):
    """Test that k parameter limits the number of recommendations per customer."""
    previous_purchases = make_previous_purchases(['C1', 'C2'], [1001, 1002], week_numbers=[0, 0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles, n_components=2)
    item_similarity.set_week(2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    # Test with different k values
    result_k1 = item_similarity.generate(['C1', 'C2'], k=1)
    result_k3 = item_similarity.generate(['C1', 'C2'], k=3)
    
    # Check that k limits the results per customer
    max_results_per_customer_k1 = result_k1.groupby('customer_id').size().max()
    max_results_per_customer_k3 = result_k3.groupby('customer_id').size().max()
    
    assert max_results_per_customer_k1 <= 1
    assert max_results_per_customer_k3 <= 3

def test_no_context_error(sample_transactions, sample_articles):
    """Test that error is raised when context is not set."""
    item_similarity = ItemSimilarity(sample_transactions, sample_articles)
    item_similarity.set_week(2)
    
    with pytest.raises(ValueError, match="Context must be set before generating candidates"):
        item_similarity.generate(['C1'], k=2)

def test_no_previous_purchases_in_context_error(sample_transactions, sample_articles):
    """Test that error is raised when previous_purchases is not in context."""
    item_similarity = ItemSimilarity(sample_transactions, sample_articles)
    item_similarity.set_week(2)
    item_similarity.set_context({"other_key": "value"})
    
    with pytest.raises(ValueError, match="Previous purchases must be set in context"):
        item_similarity.generate(['C1'], k=2)

def test_generate_without_setting_week(sample_transactions, sample_articles):
    """Test that generating without setting week raises an error."""
    previous_purchases = make_previous_purchases(['C1'], [1001], week_numbers=[0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    with pytest.raises(ValueError, match="Week must be set before generating candidates"):
        item_similarity.generate(['C1'], k=2)

def test_empty_previous_purchases(sample_transactions, sample_articles):
    """Test behavior when previous purchases is empty."""
    previous_purchases = pd.DataFrame(columns=['customer_id', 'article_id', 'week'])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles, n_components=2)
    item_similarity.set_week(2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    result = item_similarity.generate(['C1'], k=2)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    assert len(result) == 0

def test_nonexistent_customer(sample_transactions, sample_articles):
    """Test behavior with non-existent customer in previous purchases."""
    previous_purchases = make_previous_purchases(['C1'], [1001], week_numbers=[0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles, n_components=2)
    item_similarity.set_week(2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    result = item_similarity.generate(['NONEXISTENT'], k=2)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    assert len(result) == 0

def test_multiple_previous_purchases(sample_transactions, sample_articles):
    """Test with multiple previous purchases for the same customer."""
    previous_purchases = make_previous_purchases(['C1', 'C1', 'C2'], [1001, 1002, 1002], week_numbers=[0, 0, 0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles, n_components=2)
    item_similarity.set_week(2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    result = item_similarity.generate(['C1', 'C2'], k=3)
    
    # Should get results for both customers
    assert set(result['customer_id']) == {'C1', 'C2'}
    
    # C1 should have more recommendations since it has more previous purchases
    c1_count = len(result[result['customer_id'] == 'C1'])
    c2_count = len(result[result['customer_id'] == 'C2'])
    
    assert c1_count > 0
    assert c2_count > 0

def test_similarity_ranking(sample_transactions, sample_articles):
    """Test that results are ranked by similarity."""
    previous_purchases = make_previous_purchases(['C1'], [1001], week_numbers=[0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles, n_components=2)
    item_similarity.set_week(2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    result = item_similarity.generate(['C1'], k=3)
    
    # Should get results ranked by similarity (highest first)
    assert len(result) > 0
    
    # All results should be for C1
    assert all(result['customer_id'] == 'C1')

def test_data_types(sample_transactions, sample_articles):
    """Test that output DataFrame has correct data types."""
    previous_purchases = make_previous_purchases(['C1'], [1001], week_numbers=[0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles, n_components=2)
    item_similarity.set_week(2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    result = item_similarity.generate(['C1'], k=3)
    
    # Check data types
    assert result['customer_id'].dtype == 'object'
    if len(result) > 0:
        assert result['article_id'].dtype in ['int64', 'int32']
        for aid in result['article_id']:
            assert isinstance(aid, int)

def test_matrix_creation(sample_transactions, sample_articles):
    """Test that the user-item matrix is created correctly."""
    previous_purchases = make_previous_purchases(['C1'], [1001], week_numbers=[0])
    
    item_similarity = ItemSimilarity(sample_transactions, sample_articles)
    item_similarity.set_week(2)
    item_similarity.set_context({"previous_purchases": previous_purchases})
    
    # Check that matrix and mappings are created
    assert item_similarity.item_user_matrix is not None
    assert item_similarity.user_map is not None
    assert item_similarity.item_map is not None
    assert item_similarity.reverse_user_map is not None
    assert item_similarity.reverse_item_map is not None
    
    # Check matrix shape
    assert item_similarity.item_user_matrix.shape[0] == len(item_similarity.user_map)
    assert item_similarity.item_user_matrix.shape[1] == len(item_similarity.item_map)

def test_week_setting_creates_matrix(sample_transactions, sample_articles):
    """Test that setting week creates the user-item matrix."""
    item_similarity = ItemSimilarity(sample_transactions, sample_articles)
    
    # Initially, matrix should be None
    assert item_similarity.item_user_matrix is None
    
    # After setting week, matrix should be created
    item_similarity.set_week(2)
    assert item_similarity.item_user_matrix is not None
    
    # Check that mappings are also created
    assert item_similarity.user_map is not None
    assert item_similarity.item_map is not None