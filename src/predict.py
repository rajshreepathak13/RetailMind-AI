
import joblib
import pandas as pd

# Load saved pipeline
model = joblib.load("models/churn_pipeline.pkl")


def predict(customer_data):

    input_df = pd.DataFrame([customer_data])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": result,
        "churn_probability": round(float(probability), 4)
    }


# Example customer
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 55.5,
    "TotalCharges": 275.0
}

result = predict(customer)

print(result)
