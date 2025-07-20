import pytest
import pandas as pd
from datetime import datetime, timedelta
from retrieval.graph_search import GraphSearch

@pytest.fixture(scope="module")
def sample_transactions():
    """Create a sample transaction dataset for testing."""
    
    # week, customer_id, article_id
    transactions = [
        (0, "user1", "item1"),
        (0, "user1", "item2"),
        (0, "user2", "item1"),
        (0, "user2", "item2"),
        (0, "user3", "item1"),
        (0, "user3", "item3"),
        (1, "user4", "item3"),
        (1, "user4", "item4")
    ]

    user_map = {
        "user1": 0,
        "user2": 1,
        "user3": 2,
        "user4": 3
    }

    item_map = {
        "item1": 0,
        "item2": 1,
        "item3": 2,
        "item4": 3
    }

    transactions = pd.DataFrame(
        transactions,
        columns=["week", "customer_id", "article_id"]
    )

    transactions["customer_id"] = transactions["customer_id"].map(user_map)
    transactions["article_id"] = transactions["article_id"].map(item_map)

    base_date = datetime(2024, 1, 1)
    transactions["t_dat"] = pd.Series(base_date + timedelta(days=7 * week) for week in transactions["week"])

    return transactions

@pytest.fixture(scope="module")
def sample_articles():
    """Create a sample articles dataset for testing."""
    articles = [
        ("item1", "article1", "category1"),
        ("item2", "article2", "category2"),
        ("item3", "article3", "category3"),
        ("item4", "article4", "category4")
    ]

    item_map = {
        "item1": 0,
        "item2": 1,
        "item3": 2,
        "item4": 3
    }

    articles = pd.DataFrame(
        articles,
        columns=["article_id", "article_name", "category"]
    )

    articles["article_id"] = articles["article_id"].map(item_map)

    return articles 

def test_one_step(sample_transactions, sample_articles):
    """Test the graph search algorithm with one step."""

    user_map = {
        "user1": 0,
        "user2": 1,
        "user3": 2,
        "user4": 3
    }

    graph_search = GraphSearch(sample_transactions, sample_articles, max_steps=1)
    graph_search.set_week(1)
    graph_search.set_context({
        "previous_purchases": sample_transactions[["customer_id", "article_id"]]
    })
    
    recommendations = graph_search.generate([user_map["user1"]], k=10)
    assert len(recommendations) == 1
    
    recommendations = graph_search.generate([user_map["user3"]], k=10)
    assert len(recommendations) == 1

def test_two_steps(sample_transactions, sample_articles):
    """Test the graph search algorithm with two steps."""

    user_map = {
        "user1": 0,
        "user2": 1,
        "user3": 2,
        "user4": 3
    }

    graph_search = GraphSearch(sample_transactions, sample_articles, max_steps=2)
    graph_search.set_week(2)
    graph_search.set_context({
        "previous_purchases": sample_transactions[["customer_id", "article_id"]]
    })

    recommendations = graph_search.generate([user_map["user4"]], k=10)
    assert len(recommendations) == 2

def test_week_setting(sample_transactions, sample_articles):
    """Test the week setting functionality."""

    user_map = {
        "user1": 0,
        "user2": 1,
        "user3": 2,
        "user4": 3
    }

    graph_search = GraphSearch(sample_transactions, sample_articles, max_steps=2)
    graph_search.set_week(1)
    graph_search.set_context({
        "previous_purchases": sample_transactions[["customer_id", "article_id"]]
    })

    recommendations = graph_search.generate([user_map["user1"]], k=10)
    assert len(recommendations) == 1

    graph_search.set_week(2)

    recommendations = graph_search.generate([user_map["user1"]], k=10)
    assert len(recommendations) == 2

def test_ranking(sample_transactions, sample_articles):
    """Test the ranking functionality."""

    user_map = {
        "user1": 0,
        "user2": 1,
        "user3": 2,
        "user4": 3
    } 

    item_map = {
        "item1": 0,
        "item2": 1,
        "item3": 2,
        "item4": 3
    }

    graph_search = GraphSearch(sample_transactions, sample_articles, max_steps=1)
    graph_search.set_week(2)
    graph_search.set_context({
        "previous_purchases": sample_transactions[["customer_id", "article_id"]]
    })

    recommendations = graph_search.generate([user_map["user3"]], k=10)
    assert recommendations["article_id"].iloc[0] == item_map["item2"]
    assert recommendations["article_id"].iloc[1] == item_map["item4"]
