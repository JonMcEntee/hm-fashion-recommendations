import pandas as pd
from datetime import datetime, timedelta
from models.generate_recommendations import create_same_product_code

def make_transactions(customers, article_ids, week_numbers=None, base_date=None):
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

def test_basic_same_product_code():
    articles = pd.DataFrame({
        'article_id': [1, 2, 3, 4],
        'product_code': [10, 10, 20, 30]
    })
    transactions = make_transactions(['A', 'A'], [1, 1], week_numbers=[0, 1])
    same_code_finder = create_same_product_code(transactions, articles)
    result = same_code_finder(['A'], week=1, k=10)
    assert set(result['article_id']) == {1, 2}

def test_multiple_previous_purchases():
    articles = pd.DataFrame({
        'article_id': [1, 2, 3, 4, 5],
        'product_code': [10, 10, 20, 30, 20]
    })
    transactions = make_transactions(['A', 'A'], [1, 3], week_numbers=[0, 0])
    same_code_finder = create_same_product_code(transactions, articles)
    result = same_code_finder(['A'], week=1, k=10)
    assert set(result['article_id']) == {1, 2, 3, 5}

def test_duplicates_removed():
    articles = pd.DataFrame({
        'article_id': [1, 2, 3],
        'product_code': [10, 10, 10]
    })
    transactions = make_transactions(['A', 'A'], [1, 2], week_numbers=[0, 0])
    same_code_finder = create_same_product_code(transactions, articles)
    result = same_code_finder(['A'], week=1, k=10)
    assert set(result['article_id']) == {1, 2, 3}
    assert result['article_id'].duplicated().sum() == 0

def test_multiple_customers():
    articles = pd.DataFrame({
        'article_id': [1, 2, 3, 4, 5],
        'product_code': [10, 10, 20, 30, 20]
    })
    transactions = make_transactions(['A', 'B', 'B'], [1, 3, 4], week_numbers=[0, 0, 0])
    same_code_finder = create_same_product_code(transactions, articles)
    result = same_code_finder(['A', 'B'], week=1, k=10)
    a_articles = set(result[result['customer_id'] == 'A']['article_id'])
    b_articles = set(result[result['customer_id'] == 'B']['article_id'])
    assert a_articles == {1, 2}
    assert b_articles == {3, 4, 5}

def test_output_columns_and_types():
    articles = pd.DataFrame({
        'article_id': [1, 2, 3],
        'product_code': [10, 10, 20]
    })
    transactions = make_transactions(['A'], [1], week_numbers=[0])
    same_code_finder = create_same_product_code(transactions, articles)
    result = same_code_finder(['A'], week=1, k=10)
    expected_columns = {'customer_id', 'article_id'}
    assert expected_columns.issubset(set(result.columns))
    assert result['customer_id'].dtype == 'object'
    if len(result) > 0:
        assert result['article_id'].dtype in ['int64', 'int32']
        for rec in result['article_id']:
            assert isinstance(rec, int)

def test_output_is_ranked_by_time():
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
    same_code_finder = create_same_product_code(transactions, articles)
    # Use week=3 so all transactions are included as history
    result = same_code_finder(['A'], week=3, k=10)
    # The output should be sorted by recency (most recent transaction first)
    # The most recent transaction is for article 3, then 2, then 1
    expected_order = [3, 2, 1]
    actual_order = list(result['article_id'])
    assert actual_order == expected_order

def test_k_limits_number_of_recommendations():
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
    same_code_finder = create_same_product_code(transactions, articles)
    # Use week=11 so all transactions are included as history
    k = 5
    result = same_code_finder(['A'], week=11, k=k)
    # Should be at most k recommendations for customer 'A'
    assert len(result[result['customer_id'] == 'A']) <= k