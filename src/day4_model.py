
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

from xgboost import XGBClassifier
from feature_engineering import RetailFeatureEngineer


# 1. Create required folders
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


# 2. Load dataset
df = pd.read_csv(
    "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# 3. Clean target and input data
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})


# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Feature engineering
feature_engineer = RetailFeatureEngineer()

X_train_engineered = feature_engineer.fit_transform(X_train)
X_test_engineered = feature_engineer.transform(X_test)


# 6. Identify numerical and categorical columns
numeric_features = X_train_engineered.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns

categorical_features = X_train_engineered.select_dtypes(
    include=["object", "category"]
).columns


# 7. Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


# 8. Define candidate models
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / positive_count

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# 9. Build complete pipelines
pipelines = {}

for name, classifier in models.items():

    pipeline = Pipeline(steps=[
        ("feature_engineering", RetailFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(
            score_func=f_classif,
            k=40
        )),
        ("classifier", classifier)
    ])

    pipelines[name] = pipeline


# 10. Train and evaluate candidate models
results = []
trained_models = {}

for name, pipeline in pipelines.items():

    print("\nTraining:", name)

    # Cross-validation using training data only
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="recall",
        n_jobs=1
    )

    # Fit on the complete training set
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    trained_models[name] = pipeline

    results.append({
        "Model": name,
        "CV Recall": cv_scores.mean(),
        "CV Recall Std": cv_scores.std(),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Recall": recall_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    })

    print("CV Recall:", cv_scores.mean())
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(
        y_test, y_pred, zero_division=0
    ))
    print("Recall:", recall_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))


# 11. Compare with Day 3 champion
day3_path = "models/churn_v3.joblib"

if os.path.exists(day3_path):

    day3_model = joblib.load(day3_path)

    print("\nEvaluating Day 3 champion")

    day3_cv = cross_val_score(
        day3_model,
        X_train,
        y_train,
        cv=5,
        scoring="recall",
        n_jobs=1
    )

    day3_pred = day3_model.predict(X_test)

    day3_prob = day3_model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": "Day 3 Champion",
        "CV Recall": day3_cv.mean(),
        "CV Recall Std": day3_cv.std(),
        "Accuracy": accuracy_score(y_test, day3_pred),
        "Precision": precision_score(
            y_test, day3_pred, zero_division=0
        ),
        "Recall": recall_score(y_test, day3_pred),
        "ROC-AUC": roc_auc_score(y_test, day3_prob)
    })

    trained_models["Day 3 Champion"] = day3_model

else:
    print("\nDay 3 model not found. Comparing Day 4 models only.")


# 12. Create comparison report
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["CV Recall", "ROC-AUC"],
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/day4_model_comparison.csv",
    index=False
)


# 13. Select final champion using CV Recall
champion_name = results_df.iloc[0]["Model"]
champion_model = trained_models[champion_name]

print("\nSelected Champion:", champion_name)


# 14. Save production model
joblib.dump(
    champion_model,
    "models/churn_v4.joblib"
)

print("Production model saved successfully!")


# 15. Save champion information
with open("reports/day4_champion.txt", "w") as file:
    file.write(f"Champion Model: {champion_name}\n")
    file.write(
        f"Cross-validation Recall: "
        f"{results_df.iloc[0]['CV Recall']:.4f}\n"
    )
    file.write(
        f"Test Recall: {results_df.iloc[0]['Recall']:.4f}\n"
    )
    file.write(
        f"Test ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f}\n"
    )

print("\nDay 4 completed successfully!")
