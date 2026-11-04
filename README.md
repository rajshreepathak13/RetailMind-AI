
# RetailMind AI

## Project Overview

RetailMind AI is an end-to-end AI platform designed to help
retail businesses identify customers who are likely to leave
their services.

The platform uses Machine Learning techniques to predict
customer churn and help businesses take preventive actions
to improve customer retention.

## Day 1 - ML Baseline

### Objective
Build an end-to-end customer churn prediction pipeline
using the Telco Customer Churn dataset.

### Dataset
Telco Customer Churn Dataset - Kaggle

### Technologies Used
- Python
- Pandas
- Scikit-learn
- Joblib

### Project Workflow
1. Load and inspect the dataset.
2. Analyze missing values and target distribution.
3. Separate features and target variable.
4. Perform train-test splitting.
5. Encode categorical variables.
6. Scale numerical features.
7. Train a Logistic Regression classifier.
8. Evaluate model performance.
9. Save the fitted pipeline using Joblib.

### Model
Logistic Regression

### Evaluation Metrics

Update these values after running train.py.

| Metric | Score |
|---|---|
| Accuracy | Add actual score |
| Precision | Add actual score |
| Recall | Add actual score |
| ROC-AUC | Add actual score |

### Model Persistence

The trained ML pipeline is saved at:

models/churn_pipeline.pkl

### Prediction

A prediction helper is available in src/predict.py
to predict whether a customer is likely to churn.

## Future Improvements

- Improve model performance.
- Experiment with advanced ML algorithms.
- Build an API for predictions.
- Integrate the model into the RetailMind AI platform.
