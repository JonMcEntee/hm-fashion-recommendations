# H\&M Fashion Recommendations

<img src="images/H%26M-Logo.svg" width="40%" height="40%">

## Overview

This repository contains [a Jupyter Notebook](https://github.com/JonMcEntee/hm-fashion-recommendations/blob/main/HM_Fashion_Recommendations.ipynb) for building a product recommendation system. It was developed as part of the [H\&M Personalized Fashion Recommendations Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

The notebook presents an approach to predict a customer's next likely fashion purchases using:

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


## Usage

To run the code in this repository, you'll need to set up your Python environment correctly. Here's how:

1. **Create and activate the conda environment**

   First, create a new conda environment and activate it:

   ```bash
   # Create the environment
   conda create -n hm-fashion python=3.9
   
   # Activate the environment
   conda activate hm-fashion
   ```

2. **Install dependencies**

   Install all required packages from the requirements.txt file:

   ```bash
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Set up the Python path**

   Since the code is organized in modules, you need to set the PYTHONPATH to include the project root directory. This ensures Python can find all the modules correctly.

   ```bash
   # From the project root directory
   export PYTHONPATH=$(pwd)
   ```

4. **Run the code**

   After setting the PYTHONPATH, you can run any of the Python modules:

   ```bash
   # Example: Run the ALS model
   python src/models/als_model.py
   ```

5. **Launch the Jupyter notebook**

   To explore the main analysis and run the recommendation system:

   ```bash
   # Launch Jupyter notebook
   jupyter notebook
   ```

   Then navigate to and open `HM_Fashion_Recommendations.ipynb` in your browser.

   Alternatively, you can launch the notebook directly:

   ```bash
   # Launch the specific notebook
   jupyter notebook HM_Fashion_Recommendations.ipynb
   ```


## Key Findings

* Recent transaction activity is more predictive than older data
* Implicit ALS performs well for large-scale retail data with implicit feedback

## Acknowledgments

* Kaggle for organizing the competition
* H\&M Group for providing real-world retail data
