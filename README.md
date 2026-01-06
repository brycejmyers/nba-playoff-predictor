# NBA Playoff Prediction Using Machine Learning

## Overview
This project predicts whether an NBA team will qualify for the playoffs based on regular-season team statistics. Data is collected programmatically through an NBA statistics API and used to train supervised machine learning models that identify which factors are most predictive of postseason success.

The project demonstrates an end-to-end data science workflow, including data ingestion, cleaning, feature engineering, modeling, and interpretation.

## Motivation
NBA organizations make high-stakes decisions based on player and team performance data. Understanding which statistical factors contribute most to playoff qualification can inform roster construction, strategic planning, and resource allocation. This project aims to highlight how data-driven methods can support these decisions.

## Data
- **Source:** NBA statistics API (API key-based access)
- **Unit of analysis:** Team-season
- **Features include:**
  - Offensive metrics (points per game, shooting percentages, assists)
  - Defensive metrics (opponent points, steals, blocks, defensive rating)
  - Advanced and efficiency statistics
- **Target variable:** Playoff qualification (binary)

## Methods
- Data ingestion using API requests (`requests`)
- Data cleaning and feature engineering using `pandas`
- Supervised classification models:
  - Logistic Regression (baseline, interpretable)
  - Random Forest Classifier (nonlinear, feature importance)
- Model evaluation using accuracy and cross-validation
- Feature importance analysis to identify key predictors of success

## Results
Preliminary results indicate that defensive metrics—such as defensive rating and opponent points per game—are among the strongest predictors of playoff qualification. These findings align with prior academic research on NBA team success.

## Project Structure
