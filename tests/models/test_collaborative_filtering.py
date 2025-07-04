import pytest
import pandas as pd
import numpy as np
import scipy.sparse as sp
from datetime import datetime, timedelta
from src.models.collaborative_filtering import create_user_item_matrix, top_k_cosine_similarity

def make_test_transactions():
    """Create test transaction data for collaborative filtering tests."""
    base_date = datetime(2024, 1, 1)
    dates = [
        base_date,                    # Week 0
        base_date + timedelta(days=7),  # Week 1
        base_date + timedelta(days=14), # Week 2
        base_date + timedelta(days=21), # Week 3
        base_date + timedelta(days=28), # Week 4
    ]
    
    data = {
        't_dat': dates * 2,  # Repeat dates for multiple transactions
        'week': [0, 1, 2, 3, 4] * 2,
        'customer_id': ['C1', 'C1', 'C2', 'C2', 'C3'] + ['C1', 'C2', 'C3', 'C1', 'C2'],
        'article_id': [1001, 1002, 1001, 1003, 1002] + [1004, 1005, 1001, 1002, 1003]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_transactions():
    """Fixture providing sample transaction data."""
    return make_test_transactions()

def test_create_user_item_matrix_basic(sample_transactions):
    """Test basic functionality of create_user_item_matrix."""
    last_week = 4
    train_window = 5
    
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    # Check that matrix is created
    assert isinstance(matrix, sp.csr_matrix)
    assert matrix.shape[0] == len(user_map)  # Number of users
    assert matrix.shape[1] == len(item_map)  # Number of items
    
    # Check mappings
    assert len(user_map) == 3  # C1, C2, C3
    assert len(item_map) == 5  # 1001, 1002, 1003, 1004, 1005
    
    # Check reverse mappings
    assert reverse_user_map[0] in ['C1', 'C2', 'C3']
    assert reverse_item_map[0] in [1001, 1002, 1003, 1004, 1005]

def test_create_user_item_matrix_uniform_type(sample_transactions):
    """Test uniform matrix type (binary interactions)."""
    last_week = 4
    train_window = 5
    
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    # Check that matrix contains only 0s and 1s (binary)
    unique_values = set(matrix.data)
    assert unique_values.issubset({0, 1})
    
    # Check that there are some interactions (non-zero values)
    assert matrix.nnz > 0

def test_create_user_item_matrix_time_decay_type(sample_transactions):
    """Test time decay matrix type (weighted by recency)."""
    last_week = 4
    train_window = 5
    
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="time_decay"
    )
    
    # Check that matrix contains non-negative values
    assert np.all(matrix.data >= 0)
    
    # Check that there are some interactions (non-zero values)
    assert matrix.nnz > 0
    
    # Check that values are not all 1s (should be weighted)
    unique_values = set(matrix.data)
    assert len(unique_values) > 1 or (len(unique_values) == 1 and 1 not in unique_values)

def test_create_user_item_matrix_window_filtering(sample_transactions):
    """Test that the training window correctly filters transactions."""
    last_week = 3
    train_window = 2  # Only weeks 2 and 3
    
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    # Should only include transactions from weeks 2 and 3
    # Original data has transactions in weeks 0-4, but we only want 2-3
    filtered_transactions = sample_transactions[
        (sample_transactions['week'] >= last_week - train_window) &
        (sample_transactions['week'] <= last_week)
    ]
    
    # Check that matrix size reflects filtered data
    expected_users = len(filtered_transactions['customer_id'].unique())
    expected_items = len(filtered_transactions['article_id'].unique())
    
    assert matrix.shape[0] == expected_users
    assert matrix.shape[1] == expected_items

def test_create_user_item_matrix_empty_window(sample_transactions):
    """Test behavior when training window results in no data."""
    last_week = 10
    train_window = 5  # No data in weeks 5-10
    
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    # Should return empty matrix
    assert matrix.shape[0] == 0
    assert matrix.shape[1] == 0
    assert len(user_map) == 0
    assert len(item_map) == 0

def test_top_k_cosine_similarity_basic(sample_transactions):
    """Test basic functionality of top_k_cosine_similarity for items."""
    # Create user-item matrix
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    # Test with first few items
    test_items = list(item_map.keys())[:2]
    
    result = top_k_cosine_similarity(
        matrix, test_items, item_map, reverse_item_map, 
        type="item", k=3, n_components=2
    )
    
    # Check result structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'article_id', 'similarity', 'similar_object'}
    
    # Check that we get results
    assert len(result) > 0
    
    # Check that similarity scores are between -1 and 1 (cosine similarity range)
    assert np.all(result['similarity'] >= -1)
    assert np.all(result['similarity'] <= 1)

def test_top_k_cosine_similarity_users(sample_transactions):
    """Test top_k_cosine_similarity for users."""
    # Create user-item matrix
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    # Test with first few users
    test_users = list(user_map.keys())[:2]
    
    result = top_k_cosine_similarity(
        matrix, test_users, user_map, reverse_user_map, 
        type="user", k=2, n_components=2
    )
    
    # Check result structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'similarity', 'similar_object'}
    
    # Check that we get results
    assert len(result) > 0

def test_top_k_cosine_similarity_invalid_type(sample_transactions):
    """Test that invalid type raises an error."""
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    test_items = list(item_map.keys())[:1]
    
    with pytest.raises(ValueError, match="Type must be either 'item' or 'user'"):
        top_k_cosine_similarity(
            matrix, test_items, item_map, reverse_item_map, 
            type="invalid", k=3
        )

def test_top_k_cosine_similarity_k_parameter(sample_transactions):
    """Test that k parameter limits the number of results."""
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    test_items = list(item_map.keys())[:1]
    k = 2
    
    result = top_k_cosine_similarity(
        matrix, test_items, item_map, reverse_item_map, 
        type="item", k=k, n_components=2
    )
    
    # Should get at most k results per item
    max_results_per_item = result.groupby('article_id').size().max()
    assert max_results_per_item <= k

def test_top_k_cosine_similarity_missing_ids(sample_transactions):
    """Test that ValueError is raised when some IDs are not in the mapping."""
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    # Mix valid and invalid IDs
    test_items = list(item_map.keys())[:1] + [99999]  # 99999 doesn't exist
    
    with pytest.raises(ValueError, match="item 99999 not found in mapping"):
        top_k_cosine_similarity(
            matrix, test_items, item_map, reverse_item_map, 
            type="item", k=3, n_components=2
        )

def test_top_k_cosine_similarity_n_components(sample_transactions):
    """Test that n_components parameter affects the result."""
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    test_items = list(item_map.keys())[:1]
    
    # Test with different n_components
    result1 = top_k_cosine_similarity(
        matrix, test_items, item_map, reverse_item_map, 
        type="item", k=3, n_components=2
    )
    
    result2 = top_k_cosine_similarity(
        matrix, test_items, item_map, reverse_item_map, 
        type="item", k=3, n_components=1
    )
    
    # Results might be different due to different dimensionality reduction
    # But both should be valid DataFrames
    assert isinstance(result1, pd.DataFrame)
    assert isinstance(result2, pd.DataFrame)
    assert len(result1) > 0
    assert len(result2) > 0

def test_top_k_cosine_similarity_random_state(sample_transactions):
    """Test that random_state parameter provides reproducible results."""
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    test_items = list(item_map.keys())[:1]
    
    # Test with same random_state
    result1 = top_k_cosine_similarity(
        matrix, test_items, item_map, reverse_item_map, 
        type="item", k=3, n_components=2, random_state=42
    )
    
    result2 = top_k_cosine_similarity(
        matrix, test_items, item_map, reverse_item_map, 
        type="item", k=3, n_components=2, random_state=42
    )
    
    # Results should be identical with same random_state
    pd.testing.assert_frame_equal(result1, result2)

def test_matrix_transpose_for_users(sample_transactions):
    """Test that matrix is transposed correctly for user similarity."""
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    test_users = list(user_map.keys())[:1]
    
    # Test user similarity (should transpose matrix)
    result = top_k_cosine_similarity(
        matrix, test_users, user_map, reverse_user_map, 
        type="user", k=2, n_components=2
    )
    
    # Should get results for users
    assert len(result) > 0
    assert 'customer_id' in result.columns
    assert 'similar_object' in result.columns

def test_similarity_scores_range(sample_transactions):
    """Test that similarity scores are in valid range for cosine similarity."""
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    test_items = list(item_map.keys())[:1]
    
    result = top_k_cosine_similarity(
        matrix, test_items, item_map, reverse_item_map, 
        type="item", k=2, n_components=2
    )
    
    # Cosine similarity should be between -1 and 1
    assert np.all(result['similarity'] >= -1)
    assert np.all(result['similarity'] <= 1)
    
    # Most similarities should be positive (since we're dealing with positive interactions)
    positive_similarities = (result['similarity'] > 0).sum()
    assert positive_similarities > 0

def test_top_k_cosine_similarity_k_too_large(sample_transactions):
    """Test that ValueError is raised when k is greater than the number of items/users."""
    last_week = 4
    train_window = 5
    matrix, user_map, item_map, reverse_user_map, reverse_item_map = create_user_item_matrix(
        sample_transactions, last_week, train_window, matrix_type="uniform"
    )
    
    # Test with k larger than number of items
    test_items = list(item_map.keys())[:1]
    k_too_large = len(item_map)  # (k + 1) > number of items, one will be added inside top_k_cosine_simularity
    
    with pytest.raises(ValueError, match=f"k \\({k_too_large}\\) is greater than the number of items \\({len(item_map) - 1}\\) \\(not including the item itself\\)"):
        top_k_cosine_similarity(
            matrix, test_items, item_map, reverse_item_map, 
            type="item", k=k_too_large, n_components=2
        )
    
    # Test with k larger than number of users
    test_users = list(user_map.keys())[:1]
    k_too_large_users = len(user_map) + 1  # k > number of users
    
    with pytest.raises(ValueError, match=f"k \\({k_too_large_users}\\) is greater than the number of users \\({len(user_map) - 1}\\) \\(not including the user itself\\)"):
        top_k_cosine_similarity(
            matrix, test_users, user_map, reverse_user_map, 
            type="user", k=k_too_large_users, n_components=2
        )
