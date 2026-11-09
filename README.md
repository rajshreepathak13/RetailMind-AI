
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

## Day 2 - Model Comparison

### Objective
Improve the baseline churn prediction model by comparing
Decision Tree, Random Forest and Support Vector Machine.

### Algorithms Used
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

### Methodology
1. Reused Day 1 preprocessing.
2. Applied the same train-test split.
3. Trained three classification models.
4. Evaluated accuracy, precision, recall and ROC-AUC.
5. Selected the champion model based on recall.
6. Saved the champion model using Joblib.

### Model Comparison

See reports/day2_comparison.csv for actual results.

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Decision Tree | Update | Update | Update | Update |
| Random Forest | Update | Update | Update | Update |
| SVM | Update | Update | Update | Update |

### Champion Model

The model with the highest churn-class recall is selected.
ROC-AUC is used as the tie-breaker.

### Saved Models
- models/churn_pipeline.pkl - Day 1 baseline
- models/day2_champion.pkl - Day 2 champion
## Day 3 - Hyperparameter Tuning

- Technique: GridSearchCV
- Cross-validation: 5-fold
- Scoring metric: Recall
- Models: Logistic Regression, Decision Tree,
  Random Forest, SVM

### Best Model

- Model: [Your champion model]
- Best CV Recall: [Actual score]
- Test Recall: [Actual score]
- Test ROC-AUC: [Actual score]
- Best Hyperparameters: [Actual parameters]

### Bias-Variance Analysis

Training and validation recall were compared
using cross-validation. A learning curve was
generated to study model generalization.

## Day 4 - Feature Engineering and Ensemble Learning

### Feature Engineering
Created additional features:
- Tenure groups
- Average monthly spending
- Monthly-to-total charge ratio
- Contract flags
- Service count
- Internet and phone subscription flags
- Paperless billing flag

### Feature Selection
- Technique: SelectKBest
- Scoring function: ANOVA F-test
- Selected features: 40

### Ensemble Models
1. Random Forest (Bagging)
2. XGBoost (Boosting)

### Model Evaluation
- Cross-validation: 5-fold
- Primary metric: Recall
- Other metrics: Accuracy, Precision, ROC-AUC

### Day 4 Model Comparison

| Model | CV Recall | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| Day 3 Champion | Actual | Actual | Actual | Actual | Actual |
| Random Forest | Actual | Actual | Actual | Actual | Actual |
| XGBoost | Actual | Actual | Actual | Actual | Actual |

### Production Model
- Final model: [Enter selected model]
- Saved file: models/churn_v4.joblib
- Preprocessing and feature engineering are included in the saved pipeline.


