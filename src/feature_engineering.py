
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RetailFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        # Convert TotalCharges to numeric
        X["TotalCharges"] = pd.to_numeric(
            X["TotalCharges"], errors="coerce"
        )

        # 1. Tenure buckets
        X["TenureGroup"] = pd.cut(
            X["tenure"],
            bins=[-1, 12, 24, 48, 72],
            labels=["New", "Medium", "Long", "Loyal"]
        ).astype("object")

        # 2. Average monthly spending based on total charges
        X["AverageMonthlySpend"] = (
            X["TotalCharges"] / X["tenure"].replace(0, 1)
        )

        # 3. Monthly-to-total charge ratio
        X["ChargeRatio"] = (
            X["MonthlyCharges"] /
            (X["TotalCharges"] + 1)
        )

        # 4. Contract flags
        X["IsMonthToMonth"] = (
            X["Contract"] == "Month-to-month"
        ).astype(int)

        X["IsOneYearContract"] = (
            X["Contract"] == "One year"
        ).astype(int)

        X["IsTwoYearContract"] = (
            X["Contract"] == "Two year"
        ).astype(int)

        # 5. Service flags
        service_columns = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies"
        ]

        X["ServiceCount"] = sum(
            (X[col] == "Yes").astype(int)
            for col in service_columns
        )

        X["HasInternet"] = (
            X["InternetService"] != "No"
        ).astype(int)

        X["HasPhone"] = (
            X["PhoneService"] == "Yes"
        ).astype(int)

        X["PaperlessBillingFlag"] = (
            X["PaperlessBilling"] == "Yes"
        ).astype(int)

        return X
