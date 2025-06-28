"""
This script generates training samples for ranking models in the H&M Fashion Recommendations project.
"""
import pandas as pd
from tqdm import tqdm
from src.models.generate_recommendations import create_recommendation_generator


def generate_recommendations():
    """
    Generate candidate recommendations for each customer and week, and save them to a CSV file.
    This function loads transaction and article data, creates a recommender, and iterates over weeks to generate recommendations.
    For each week, it samples a subset of recommendations and appends them to a CSV file.
    """
    print("Loading data...")
    directory = "data/"
    transactions = pd.read_csv(directory + "transactions_train.csv", parse_dates=['t_dat'])
    articles = pd.read_csv(directory + "articles.csv")

    print("Creating recommender...")
    recommender = create_recommendation_generator(transactions, articles)

    # Calculate the week number for each transaction
    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['week'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7

    print("Generating recommendations...")
    first_week = 50  # Start generating from week 50
    num_samples = 0
    for week in tqdm(range(first_week, 105)):
        # Get unique customers for the current week
        customers = transactions[transactions['week'] == week]['customer_id'].unique().tolist()
        # Generate recommendations for these customers
        recommendations = recommender(customers, week=week, k=12).sample(2_000_000)
        recommendations["week"] = week
        num_samples += len(recommendations)
        # Save recommendations to CSV (append after the first week)
        if week == first_week:
            recommendations.to_csv("data/recommendations.csv", index=False)
        else:
            recommendations.to_csv("data/recommendations.csv", mode="a", header=False, index=False)

    print(f"Recommendations saved to recommendations.csv with {num_samples} samples")

def generate_labels():
    """
    Generate labeled data for ranking models by combining positive and negative samples.
    Positive samples are actual purchases; negative samples are recommendations not purchased.
    Saves the labeled dataset to a CSV file.
    """
    print("Loading data...")
    transactions = pd.read_csv("data/transactions_train.csv", parse_dates=['t_dat'])

    # Calculate the week number for each transaction
    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['7d'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7

    # Extract positive samples (actual purchases)
    positive_samples = transactions[['article_id', 'customer_id', '7d']]\
        .drop_duplicates()

    print("Loading recommendations...")
    recommendations = pd.read_csv("data/recommendations.csv")\
        .rename(columns={'week': '7d', 'recommendation': 'article_id'})

    print("Generating negative samples...")
    # Merge recommendations with positive samples to identify negatives
    negative_samples = pd.merge(
        recommendations,
        positive_samples,
        on=['customer_id', '7d', 'article_id'],
        how='left',
        indicator=True
    )

    # Filter to only keep samples not in positive (i.e., negatives)
    negative_samples = negative_samples[negative_samples['_merge'] == 'left_only']

    # Drop the merge indicator column
    negative_samples = negative_samples.drop('_merge', axis=1)

    # Assign labels
    positive_samples["label"] = 1
    negative_samples['label'] = 0

    print("Concatenating positive and negative samples...")
    # Combine positive and negative samples
    samples = pd.concat([positive_samples, negative_samples])

    print(f"Saving {len(samples)} samples...")
    samples.to_csv("data/labeled_recommendations.csv", index=False)


if __name__ == "__main__":
    generate_labels()