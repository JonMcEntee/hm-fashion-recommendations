import pandas as pd
import numpy as np
import pytest
from features.feature_generator import FeatureGenerator, Feature, DerivativeFeature

def mock_data():
    transactions = pd.DataFrame({
        't_dat': pd.to_datetime(['2021-09-01', '2021-09-08', '2021-09-15', '2021-09-22']),
        'article_id': [1, 1, 2, 2],
        'customer_id': ['A', 'A', 'B', 'B'],
        'price': [10, 20, 30, 40],
        'sales_channel_id': [1, 2, 1, 2]
    })
    articles = pd.DataFrame({
        'article_id': [1, 2],
        'prod_name': ['Shirt', 'Pants'],
        'product_group_name': ['Tops', 'Bottoms']
    })
    customers = pd.DataFrame({
        'customer_id': ['A', 'B'],
        'age': [25, 35]
    })
    return transactions, articles, customers

def test_feature_generator_fit_and_transform():
    transactions, articles, customers = mock_data()
    features = {
        'article_sales_count': Feature(agg='count', column=None, by=('article_id',)),
        'customer_sales_count': Feature(agg='count', column=None, by=('customer_id',))
    }
    derivative_features = {}
    derivative_functions = {}
    fg = FeatureGenerator(features, derivative_features, derivative_functions)
    fg.fit(transactions, articles, customers)
    assert ('article_id',) in fg.feature_dictionary
    assert ('customer_id',) in fg.feature_dictionary
    # Test transform
    recs = pd.DataFrame({'article_id': [1, 2], 'customer_id': ['A', 'B']})
    result = fg.transform(recs)
    assert 'article_sales_count' in result.columns
    assert 'customer_sales_count' in result.columns
    assert not result.isnull().any().any()

def test_feature_generator_verbose_flag(capsys):
    transactions, articles, customers = mock_data()
    features = {
        'article_sales_count': Feature(agg='count', column=None, by=('article_id',))
    }
    fg = FeatureGenerator(features, {}, {})
    fg.fit(transactions, articles, customers, verbose=True)
    captured = capsys.readouterr()
    assert 'Calculating' in captured.out or 'Merging' in captured.out
    # Now test with verbose=False
    fg = FeatureGenerator(features, {}, {})
    fg.fit(transactions, articles, customers, verbose=False)
    captured = capsys.readouterr()
    assert captured.out == ''

def test_feature_generator_count_aggregation():
    transactions, articles, customers = mock_data()
    features = {
        'article_sales_count': Feature(agg='count', column=None, by=('article_id',)),
    }
    fg = FeatureGenerator(features, {}, {})
    fg.fit(transactions, articles, customers)
    recs = pd.DataFrame({'article_id': [1, 2]})
    result = fg.transform(recs)
    # article_id 1 appears twice, article_id 2 appears twice
    assert result.loc[result['article_id'] == 1, 'article_sales_count'].iloc[0] == 2
    assert result.loc[result['article_id'] == 2, 'article_sales_count'].iloc[0] == 2

def test_feature_generator_sum_aggregation():
    transactions, articles, customers = mock_data()
    features = {
        'article_price_sum': Feature(agg='sum', column='price', by=('article_id',)),
    }
    fg = FeatureGenerator(features, {}, {})
    fg.fit(transactions, articles, customers)
    recs = pd.DataFrame({'article_id': [1, 2]})
    result = fg.transform(recs)
    # article_id 1: price 10+20=30, article_id 2: price 30+40=70
    assert result.loc[result['article_id'] == 1, 'article_price_sum'].iloc[0] == 30
    assert result.loc[result['article_id'] == 2, 'article_price_sum'].iloc[0] == 70

def test_feature_generator_derivative_feature():
    transactions, articles, customers = mock_data()
    features = {
        'article_sales_count': Feature(agg='count', column=None, by=('article_id',)),
        'article_price_sum': Feature(agg='sum', column='price', by=('article_id',)),
    }
    derivative_features = {
        'price_per_sale': DerivativeFeature(
            feature_1='article_price_sum',
            feature_2='article_sales_count',
            function='divide'
        )
    }
    derivative_functions = {
        'divide': lambda x, y: x / y
    }
    fg = FeatureGenerator(features, derivative_features, derivative_functions)
    fg.fit(transactions, articles, customers)
    recs = pd.DataFrame({'article_id': [1, 2]})
    result = fg.transform(recs)
    # article_id 1: price_sum=30, sales_count=2, price_per_sale=15
    # article_id 2: price_sum=70, sales_count=2, price_per_sale=35
    assert result.loc[result['article_id'] == 1, 'price_per_sale'].iloc[0] == 15
    assert result.loc[result['article_id'] == 2, 'price_per_sale'].iloc[0] == 35 