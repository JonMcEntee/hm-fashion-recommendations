# H\&M Fashion Recommendations

<img src="H%26M-Logo.svg" width="40%" height="40%">

## Overview

This repository contains [a Jupyter Notebook](https://github.com/JonMcEntee/hm-fashion-recommendations/blob/main/H%26M_Fashion_Recommendations.ipynb) for building a product recommendation system. It was developed as part of the [H\&M Personalized Fashion Recommendations Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

The notebook presents an approach to predict a customer’s next likely fashion purchases using:

* **Collaborative filtering** with the Implicit Alternating Least Squares (ALS) algorithm
* **Ranking by Gradient Boosted Trees** using the LightGBM Framework

Key datasets used:

* `transactions_train.csv` – historical purchase data
* `articles.csv` – metadata for fashion items
* `customers.csv` – anonymized customer details

## Key Features

1. **Exploratory Data Analysis (EDA)**

   * Customer and article statistics
   * Weekly transaction aggregation
   * Time-based user behavior exploration

2. **ALS Modeling Approach**

   * Implicit ALS training on a user-item interaction matrix
   * Use of transaction frequency as implicit feedback
   * Tuning latent factors and regularization parameters

3. **LightGBM Ranker Approach**

   * Generates recommendation based on simple heuristics
   * Produces trainable features on the data
   * Ranks using a Gradient Boosted Tree algorithm

## Requirements

The notebook requires the following Python packages:

* pandas
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

## Key Findings

* Recent transaction activity is more predictive than older data
* Implicit ALS performs well for large-scale retail data with implicit feedback

## Acknowledgments

* Kaggle for organizing the competition
* H\&M Group for providing real-world retail data
