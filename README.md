# H&M Fashion Recommendations

<img src="images/H%26M-Logo.svg" width="40%" height="40%">

## Overview

This repository provides a modular, production-ready codebase for building and evaluating personalized product recommendation systems, developed for the [H&M Personalized Fashion Recommendations Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

The project supports:
- **Collaborative Filtering** with Implicit ALS
- **Learning-to-Rank** with LightGBM (LambdaRank)
- **Automated Feature Engineering**
- **Comprehensive Evaluation & Hyperparameter Tuning**

A [Jupyter Notebook](https://github.com/JonMcEntee/hm-fashion-recommendations/blob/main/HM_Fashion_Recommendations.ipynb) (`HM_Fashion_Recommendations.ipynb`) demonstrates the full workflow, from EDA to model training and evaluation.

## Project Structure

```
hm-fashion-recommendations/
├── src/
│   ├── models/                # ALS, LightGBM, baselines, candidate generation
│   ├── features/              # Feature engineering and transformation
│   ├── evaluation/            # MAP@12 and APK metrics
│   ├── optimization/          # Grid search for ALS and LightGBM
│   └── ...
├── data/                     # (Not included) Large CSVs from Kaggle
├── saved_models/              # Pickled models and feature generators
├── results/                   # Model evaluation results
├── tests/                     # Unit tests for features and recommenders
├── images/                    # Project images and logo
├── HM_Fashion_Recommendations.ipynb  # Main notebook
├── requirements.txt           # Python dependencies
└── README.md
```

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/JonMcEntee/hm-fashion-recommendations.git
cd hm-fashion-recommendations
```

2. **Set up your environment**

```bash
conda create -n hm-fashion python=3.9
conda activate hm-fashion
pip install -r requirements.txt
```

3. **Set the Python path**

```bash
export PYTHONPATH=$(pwd)
```

## Data

- Download the competition data from Kaggle and place the CSVs in the `data/` directory.
- Key files: `transactions_train.csv`, `articles.csv`, `customers.csv`
- Large files (not included in repo): see [Kaggle competition page](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

## Usage

### 1. **Jupyter Notebook**

The main workflow is demonstrated in:

```bash
jupyter notebook HM_Fashion_Recommendations.ipynb
```

### 2. **Run Models from Command Line**

- **ALS Model:**
  ```bash
  python src/models/als_model.py
  ```
- **LightGBM Ranker:**
  ```bash
  python src/models/ranker_model.py
  ```
- **Grid Search (ALS):**
  ```bash
  python src/optimization/als_grid_search.py
  ```
- **Grid Search (Ranker):**
  ```bash
  python src/optimization/ranker_grid_search.py
  ```

## Models & Features

- **ALS Collaborative Filtering:**
  - Trains on user-item interaction matrix with time-decayed implicit feedback
  - Handles cold start with temporal popularity baseline
- **LightGBM LambdaRank:**
  - Ranks candidate recommendations using engineered features
  - Features include: purchase counts, product group stats, customer demographics, and derivatives
- **Baselines:**
  - Global popularity, temporal popularity, random recommender
- **Candidate Generation:**
  - Combines various simple heuristics and collorative approaches to generate potential recommendation candidates.
- **Feature Engineering:**
  - Automated via `FeatureGenerator` (see `src/features/feature_generator.py`)

## Evaluation

- **Metrics:**
  - Mean Average Precision at 12 (MAP@12)
  - Average Precision at k (APK)
- **Evaluation scripts:**
  - See `src/evaluation/metrics.py`

## Hyperparameter Tuning

- **ALS:**
  - Grid search over factors, regularization, iterations, alpha
  - See `src/optimization/als_grid_search.py`
- **LightGBM Ranker:**
  - Grid search over n_estimators, boosting type, min_child_samples, learning_rate
  - See `src/optimization/ranker_grid_search.py`

## Testing

- Unit tests for feature engineering and candidate generation in `tests/`
- Run with:
  ```bash
  pytest tests/
  ```

## Acknowledgments

- Kaggle for organizing the competition
- H&M Group for providing real-world retail data
