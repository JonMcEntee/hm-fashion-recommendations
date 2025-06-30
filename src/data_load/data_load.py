import pandas as pd

def load_data(
    transactions_path: str = "data/transactions_train.csv",
    articles_path: str = "data/articles.csv",
    customers_path: str = "data/customers.csv",
    parse_dates: bool = True
):
    """
    Load transactions, articles, and optionally customers data from CSV files.

    Args:
        transactions_path (str): Path to the transactions CSV file.
        articles_path (str): Path to the articles CSV file.
        customers_path (str, optional): Path to the customers CSV file. If None, customers will not be loaded.
        parse_dates (bool): Whether to parse dates in the transactions file.

    Returns:
        tuple: (transactions_df, articles_df, customers_df or None)
    """
    transactions = pd.read_csv(transactions_path, parse_dates=["t_dat"] if parse_dates else None)
    articles = pd.read_csv(articles_path)
    customers = pd.read_csv(customers_path) if customers_path is not None else None
    
    customer_map = {customer: i for i, customer in enumerate(customers['customer_id'].unique())}
    reverse_customer_map = {i: customer for customer, i in customer_map.items()}

    transactions['customer_id'] = transactions['customer_id'].map(customer_map)
    customers['customer_id'] = customers['customer_id'].map(customer_map)

    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['week'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7

    return transactions, articles, customers, customer_map, reverse_customer_map