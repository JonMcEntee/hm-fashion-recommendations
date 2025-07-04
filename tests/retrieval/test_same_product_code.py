import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.retrieval.same_product_code import SameProductCode

def make_transactions(customers, article_ids, week_numbers=None, base_date=None):
    """Helper function to create test transaction data."""
    if base_date is None:
        base_date = datetime(2020, 9, 1)
    if week_numbers is None:
        week_numbers = [0 for _ in customers]
    t_dat = [base_date + timedelta(days=7 * w) for w in week_numbers]
    return pd.DataFrame({
        'customer_id': customers,
        'article_id': article_ids,
        't_dat': t_dat,
        'week': week_numbers
    })

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
def sample_articles():
    """Create a sample articles dataset for testing."""
    return pd.DataFrame({
        'article_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'product_code': [10, 10, 20, 30, 20, 40, 50, 60, 70, 80],
        'product_type': ['Type1', 'Type1', 'Type2', 'Type3', 'Type2', 'Type4', 'Type5', 'Type6', 'Type7', 'Type8']
    })

def test_basic_same_product_code(sample_articles):
    """Test basic functionality of the same product code candidate generator."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3, 4],
        'product_code': [10, 10, 20, 30]
    })
    transactions = make_transactions(['A', 'A'], [1, 1], week_numbers=[0, 1])
    previous_purchases = make_previous_purchases(['A'], [1], week_numbers=[0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    result = same_code_finder.generate(['A'], k=10)
    
    # Should return articles 1 and 2 (same product code as article 1)
    assert set(result['article_id']) == {1, 2}

def test_multiple_previous_purchases(sample_articles):
    """Test with multiple previous purchases for the same customer."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3, 4, 5],
        'product_code': [10, 10, 20, 30, 20]
    })
    transactions = make_transactions(['A', 'A'], [1, 3], week_numbers=[0, 0])
    previous_purchases = make_previous_purchases(['A', 'A'], [1, 3], week_numbers=[0, 0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    result = same_code_finder.generate(['A'], k=10)
    
    # Should return articles 1, 2 (same product code as article 1) and 3, 5 (same product code as article 3)
    assert set(result['article_id']) == {1, 2, 3, 5}

def test_duplicates_removed(sample_articles):
    """Test that duplicate recommendations are removed."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3],
        'product_code': [10, 10, 10]
    })
    transactions = make_transactions(['A', 'A'], [1, 2], week_numbers=[0, 0])
    previous_purchases = make_previous_purchases(['A', 'A'], [1, 2], week_numbers=[0, 0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    result = same_code_finder.generate(['A'], k=10)
    
    # Should return all three articles (same product code) without duplicates
    assert set(result['article_id']) == {1, 2, 3}
    assert result['article_id'].duplicated().sum() == 0

def test_multiple_customers(sample_articles):
    """Test retrieving recommendations for multiple customers."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3, 4, 5],
        'product_code': [10, 10, 20, 30, 20]
    })
    transactions = make_transactions(['A', 'B', 'B'], [1, 3, 4], week_numbers=[0, 0, 0])
    previous_purchases = make_previous_purchases(['A', 'B', 'B'], [1, 3, 4], week_numbers=[0, 0, 0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    result = same_code_finder.generate(['A', 'B'], k=10)
    
    # Check specific customer recommendations
    a_articles = set(result[result['customer_id'] == 'A']['article_id'])
    b_articles = set(result[result['customer_id'] == 'B']['article_id'])
    
    # A: articles 1, 2 (same product code as article 1)
    assert a_articles == {1, 2}
    # B: articles 3, 4, 5 (same product codes as articles 3 and 4)
    assert b_articles == {3, 4, 5}

def test_output_columns_and_types(sample_articles):
    """Test that output DataFrame has correct columns and data types."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3],
        'product_code': [10, 10, 20]
    })
    transactions = make_transactions(['A'], [1], week_numbers=[0])
    previous_purchases = make_previous_purchases(['A'], [1], week_numbers=[0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    result = same_code_finder.generate(['A'], k=10)
    
    # Check column names
    expected_columns = {'customer_id', 'article_id'}
    assert expected_columns.issubset(set(result.columns))
    
    # Check data types
    assert result['customer_id'].dtype == 'object'
    if len(result) > 0:
        assert result['article_id'].dtype in ['int64', 'int32']
        for rec in result['article_id']:
            assert isinstance(rec, int)

def test_output_is_ranked_by_time(sample_articles):
    """Test that output is ranked by time (most recent first)."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3],
        'product_code': [10, 10, 10]
    })
    # Create transactions for the same customer and product code, different weeks
    transactions = make_transactions(
        ['A', 'A', 'A'],
        [1, 2, 3],
        week_numbers=[0, 1, 2]
    )
    previous_purchases = make_previous_purchases(['A', 'A', 'A'], [1, 2, 3], week_numbers=[0, 1, 2])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(3)  # Use week=3 so all transactions are included as history
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    result = same_code_finder.generate(['A'], k=10)
    
    # The output should be sorted by recency (most recent transaction first)
    # The most recent transaction is for article 3, then 2, then 1
    expected_order = [3, 2, 1]
    actual_order = list(result['article_id'])
    assert actual_order == expected_order

def test_k_limits_number_of_recommendations(sample_articles):
    """Test that k parameter limits the number of recommendations per customer."""
    articles = pd.DataFrame({
        'article_id': list(range(1, 11)),
        'product_code': [10] * 10
    })
    # 10 transactions for the same customer, different articles
    transactions = make_transactions(
        ['A'] * 10,
        list(range(1, 11)),
        week_numbers=list(range(10))
    )
    previous_purchases = make_previous_purchases(
        ['A'] * 10,
        list(range(1, 11)),
        week_numbers=list(range(10))
    )
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(11)  # Use week=11 so all transactions are included as history
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    k = 5
    result = same_code_finder.generate(['A'], k=k)
    
    # Should be at most k recommendations for customer 'A'
    assert len(result[result['customer_id'] == 'A']) <= k

def test_no_context_error(sample_articles):
    """Test that error is raised when context is not set."""
    articles = pd.DataFrame({
        'article_id': [1, 2],
        'product_code': [10, 10]
    })
    transactions = make_transactions(['A'], [1], week_numbers=[0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    
    with pytest.raises(ValueError, match="Context must be set before generating candidates"):
        same_code_finder.generate(['A'], k=10)

def test_no_previous_purchases_in_context_error(sample_articles):
    """Test that error is raised when previous_purchases is not in context."""
    articles = pd.DataFrame({
        'article_id': [1, 2],
        'product_code': [10, 10]
    })
    transactions = make_transactions(['A'], [1], week_numbers=[0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    same_code_finder.set_context({"other_key": "value"})
    
    with pytest.raises(ValueError, match="Previous purchases must be set in context"):
        same_code_finder.generate(['A'], k=10)

def test_generate_without_setting_week(sample_articles):
    """Test that generating without setting week raises an error."""
    articles = pd.DataFrame({
        'article_id': [1, 2],
        'product_code': [10, 10]
    })
    transactions = make_transactions(['A'], [1], week_numbers=[0])
    previous_purchases = make_previous_purchases(['A'], [1], week_numbers=[0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    with pytest.raises(ValueError, match="Week must be set before generating candidates"):
        same_code_finder.generate(['A'], k=10)

def test_empty_previous_purchases(sample_articles):
    """Test behavior when previous purchases is empty."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3],
        'product_code': [10, 10, 20]
    })
    transactions = make_transactions(['A'], [1], week_numbers=[0])
    previous_purchases = pd.DataFrame(columns=['customer_id', 'article_id', 'week'])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    result = same_code_finder.generate(['A'], k=10)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    assert len(result) == 0

def test_nonexistent_customer(sample_articles):
    """Test behavior with non-existent customer in previous purchases."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3],
        'product_code': [10, 10, 20]
    })
    transactions = make_transactions(['A'], [1], week_numbers=[0])
    previous_purchases = make_previous_purchases(['NONEXISTENT'], [1], week_numbers=[0])
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_week(1)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    result = same_code_finder.generate(['A'], k=10)
    
    # Should return empty DataFrame with correct structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'customer_id', 'article_id'}
    assert len(result) == 0

def test_week_filtering(sample_articles):
    """Test that only transactions before the specified week are considered."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3, 4],
        'product_code': [10, 10, 20, 20]
    })
    # Create transactions across multiple weeks
    transactions = make_transactions(
        ['A', 'A', 'A', 'A'],
        [1, 2, 3, 4],
        week_numbers=[0, 1, 2, 3]
    )
    previous_purchases = make_previous_purchases(
        ['A', 'A', 'A', 'A'],
        [1, 2, 3, 4],
        week_numbers=[0, 1, 2, 3]
    )
    
    same_code_finder = SameProductCode(transactions, articles)
    same_code_finder.set_context({"previous_purchases": previous_purchases})
    
    # Test with week=2 (should only include weeks 0 and 1)
    same_code_finder.set_week(2)
    result_week2 = same_code_finder.generate(['A'], k=10)
    
    # Should only include articles from weeks 0 and 1 (1, 2, 3)
    assert set(result_week2['article_id']) == {1, 2, 3, 4}
    
    # Test with week=4 (should include all weeks)
    same_code_finder.set_week(4)
    result_week4 = same_code_finder.generate(['A'], k=10)
    
    # Should include all articles
    assert set(result_week4['article_id']) == {1, 2, 3, 4}

def test_similar_articles_mapping(sample_articles):
    """Test that the similar_articles mapping is correctly built."""
    articles = pd.DataFrame({
        'article_id': [1, 2, 3, 4, 5],
        'product_code': [10, 10, 20, 20, 30]
    })
    transactions = make_transactions(['A'], [1], week_numbers=[0])
    
    same_code_finder = SameProductCode(transactions, articles)
    
    # Check that similar_articles mapping is correct
    assert same_code_finder.similar_articles[1] == [1, 2]  # product_code 10
    assert same_code_finder.similar_articles[2] == [1, 2]  # product_code 10
    assert same_code_finder.similar_articles[3] == [3, 4]  # product_code 20
    assert same_code_finder.similar_articles[4] == [3, 4]  # product_code 20
    assert same_code_finder.similar_articles[5] == [5]     # product_code 30

def test_weighted_transactions_structure(sample_articles):
    """Test that weighted_transactions has the expected structure."""
    articles = pd.DataFrame({
        'article_id': [1, 2],
        'product_code': [10, 10]
    })
    transactions = make_transactions(['A', 'A'], [1, 1], week_numbers=[0, 1])
    
    same_code_finder = SameProductCode(transactions, articles)
    
    # Check that weighted_transactions has expected columns
    expected_columns = {'article_id', 'week', 'time_decay', 'cumulative_weight'}
    assert expected_columns.issubset(set(same_code_finder.weighted_transactions.columns))
    
    # Check that it's grouped by article_id and week
    assert len(same_code_finder.weighted_transactions) > 0
    assert 'customer_id' not in same_code_finder.weighted_transactions.columns

def test_time_decay_calculation(sample_articles):
    """Test that time decay is calculated correctly."""
    articles = pd.DataFrame({
        'article_id': [1, 2],
        'product_code': [10, 10]
    })
    # Create transactions with known weeks
    transactions = make_transactions(['A', 'A'], [1, 1], week_numbers=[0, 2])
    
    same_code_finder = SameProductCode(transactions, articles)
    
    # Check that time decay decreases with older weeks
    weighted_df = same_code_finder.weighted_transactions
    week_0_decay = weighted_df[weighted_df['week'] == 0]['time_decay'].iloc[0]
    week_2_decay = weighted_df[weighted_df['week'] == 2]['time_decay'].iloc[0]
    
    # Week 2 should have higher decay (more recent) than week 0
    assert week_2_decay > week_0_decay

def test_cumulative_weight_calculation(sample_articles):
    """Test that cumulative weight is calculated correctly."""
    articles = pd.DataFrame({
        'article_id': [1],
        'product_code': [10]
    })
    # Create multiple transactions for the same article across weeks
    transactions = make_transactions(['A', 'A', 'A'], [1, 1, 1], week_numbers=[0, 1, 2])
    
    same_code_finder = SameProductCode(transactions, articles)
    
    weighted_df = same_code_finder.weighted_transactions
    # Sort by week to check cumulative progression
    weighted_df = weighted_df.sort_values('week')
    
    # Cumulative weight should increase over time
    cumulative_weights = weighted_df['cumulative_weight'].tolist()
    assert cumulative_weights[0] <= cumulative_weights[1] <= cumulative_weights[2]
