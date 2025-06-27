import pandas as pd
from tqdm import tqdm
from src.models.generate_recommendations import create_recommendation_generator


def main():
    print("Loading data...")
    directory = "data/"
    transactions = pd.read_csv(directory + "transactions_train.csv", parse_dates=['t_dat'])
    articles = pd.read_csv(directory + "articles.csv")

    print("Creating recommender...")
    recommender = create_recommendation_generator(transactions, articles)

    last_week = (transactions.t_dat.max() - transactions.t_dat.min()).days // 7
    transactions['week'] = last_week - (transactions.t_dat.max() - transactions.t_dat).dt.days // 7

    print("Generating recommendations...")
    for week in tqdm(range(1, 105)):
        customers = transactions[transactions['week'] == week]['customer_id'].unique().tolist()
        recommendations = recommender(customers, week=week, k=12).sample(400000)
        recommendations["week"] = week
        if week == 1:
            recommendations.to_csv("data/recommendations.csv", index=False)
        else:
            recommendations.to_csv("data/recommendations.csv", mode="a", header=False, index=False)

    print("Recommendations saved to recommendations.csv")


if __name__ == "__main__":
    main() 