# H\&M Fashion Recommendations

<img src="https://upload.wikimedia.org/wikipedia/commons/5/53/H%26M-Logo.svg" width="40%" height="40%">

## Overview

This repository contains [a Jupyter Notebook](https://github.com/JonMcEntee/hm-fashion-recommendations/blob/main/H%26M_Fashion_Recommendations.ipynb) for building a product recommendation system using collaborative filtering. It was developed as part of the [H\&M Personalized Fashion Recommendations Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

The notebook presents an approach to predict a customer’s next likely fashion purchases using:

* **Collaborative filtering** with the Implicit Alternating Least Squares (ALS) algorithm
* **Sparse matrix representations** of user-item interactions
* **Temporal filtering** of transactions to prioritize recent activity

Key datasets used:

* `transactions_train.csv` – historical purchase data
* `articles.csv` – metadata for fashion items
* `customers.csv` – anonymized customer details

## Key Features

1. **Exploratory Data Analysis (EDA)**

   * Customer and article statistics
   * Weekly transaction aggregation
   * Time-based user behavior exploration

2. **Modeling Approach**

   * Implicit ALS training on a user-item interaction matrix
   * Use of transaction frequency as implicit feedback
   * Tuning latent factors and regularization parameters

3. **Recommendation System**

   * Ranked product predictions per user
   * Filtering of already-purchased items from recommendations
   * Weekly-based train/test split for offline evaluation

## Requirements

The notebook requires the following Python packages:

* pandas,
* numpy
* scipy
* scikit-learn
* matplotlib
* seaborn
* implicit

Install dependencies with:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn implicit
```

## Usage

1. Upload the H\&M dataset to your Google Drive
2. Mount Google Drive in Colab
3. Run the notebook cells sequentially to:

   * Load and preprocess data
   * Train an ALS recommendation model
   * Generate top-N item predictions per user

## Key Findings

* Implicit ALS performs well for large-scale retail data with implicit feedback
* Recent transaction activity is more predictive than older data
* Matrix factorization enables scalable personalization without needing item content

## Acknowledgments

* Kaggle for organizing the competition
* H\&M Group for providing real-world retail data
* The `implicit` library team for the ALS implementation
