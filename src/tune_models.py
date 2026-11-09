
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    learning_curve
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# 1. Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# 2. Clean dataset
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Identify feature types
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

# 4. Preprocessing
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

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Define candidate models and parameter grids
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=2000),
        {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=42),
        {
            "classifier__max_depth": [3, 5, 10, None],
            "classifier__min_samples_split": [2, 10, 20],
            "classifier__class_weight": [None, "balanced"]
        }
    ),

    "Random Forest": (
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_split": [2, 10],
            "classifier__class_weight": [None, "balanced"]
        }
    ),

    "SVM": (
        SVC(probability=True, random_state=42),
        {
            "classifier__C": [0.1, 1, 10],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__class_weight": [None, "balanced"]
        }
    )
}

# 7. Tune models using GridSearchCV
results = []
best_models = {}

for name, (classifier, param_grid) in models.items():

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_models[name] = best_model

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Best CV Recall": grid.best_score_,
        "Train Recall": grid.cv_results_[
            "mean_train_score"
        ][grid.best_index_],
        "Test Accuracy": accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "Test Recall": recall_score(y_test, y_pred),
        "Test ROC-AUC": roc_auc_score(y_test, y_prob),
        "Best Parameters": grid.best_params_
    })

    print("\nModel:", name)
    print("Best Parameters:", grid.best_params_)
    print("Best CV Recall:", grid.best_score_)

# 8. Compare all tuned models
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Best CV Recall", "Test ROC-AUC"],
    ascending=False
)

print("\nFinal Model Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/day3_tuning_results.csv",
    index=False
)

# 9. Select champion model
champion_name = results_df.iloc[0]["Model"]
champion_model = best_models[champion_name]

joblib.dump(
    champion_model,
    "models/churn_v3.joblib"
)

print("\nChampion Model:", champion_name)
print("Saved as models/churn_v3.joblib")

# 10. Generate learning curve for champion
train_sizes, train_scores, validation_scores = learning_curve(
    champion_model,
    X_train,
    y_train,
    cv=5,
    scoring="recall",
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
validation_mean = validation_scores.mean(axis=1)

plt.figure(figsize=(8, 5))

plt.plot(train_sizes, train_mean, marker="o", label="Training Recall")
plt.plot(train_sizes, validation_mean, marker="o", label="Validation Recall")

plt.xlabel("Training Examples")
plt.ylabel("Recall")
plt.title("Learning Curve - Champion Model")
plt.legend()
plt.grid(True)

plt.savefig("reports/day3_learning_curve.png")
plt.show()
