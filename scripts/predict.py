"""
predict.py - score new bank customers with the saved churn model

How to use (run from the main project folder):

    python scripts/predict.py
        -> scores one example customer (written below)

    python scripts/predict.py path/to/new_customers.csv
        -> scores every row of the CSV and saves <file>_scored.csv

The CSV needs the same columns as the original dataset:
credit_score, country, gender, age, tenure, balance, products_number,
credit_card, active_member, estimated_salary   (customer_id is optional)
"""

import os
import sys
import pickle
import pandas as pd

# paths (works no matter which folder you run it from)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(PROJECT_DIR, "models", "scaler.pkl")

# same settings as notebook 05
THRESHOLD = 0.3
NUM_COLS = ['age', 'credit_score', 'balance', 'estimated_salary']
# column order must be exactly the same as when the model was trained
MODEL_COLUMNS = ['credit_score', 'gender', 'age', 'tenure', 'balance', 'products_number',
                 'credit_card', 'active_member', 'estimated_salary',
                 'country_Germany', 'country_Spain']


def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        sys.exit("Model files not found. Run notebooks/05_model_export.ipynb first.")
    except Exception as e:
        # usually happens when scikit-learn version is different from the one used to save the model
        sys.exit(f"Could not load model ({e}).\n"
                 "Fix: re-run notebooks/05_model_export.ipynb on this computer to save the model again.")
    return model, scaler


def prepare(df):
    """Do the same feature engineering as the notebooks."""
    data = df.copy()
    data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

    # one-hot country by hand, because a small new file may not have all 3 countries
    data['country_Germany'] = (data['country'] == 'Germany').astype(int)
    data['country_Spain'] = (data['country'] == 'Spain').astype(int)

    data = data[MODEL_COLUMNS]
    return data


def risk_band(p):
    if p >= 0.6:
        return 'High'
    elif p >= THRESHOLD:
        return 'Medium'
    else:
        return 'Low'


def predict(df, model, scaler):
    X = prepare(df)
    X[NUM_COLS] = scaler.transform(X[NUM_COLS])
    result = df.copy()
    result['churn_probability'] = model.predict_proba(X)[:, 1].round(3)
    result['predicted_churn'] = (result['churn_probability'] >= THRESHOLD).astype(int)
    result['risk_band'] = result['churn_probability'].apply(risk_band)
    return result


if __name__ == "__main__":
    model, scaler = load_model()

    if len(sys.argv) > 1:
        # score a CSV file
        input_path = sys.argv[1]
        new_customers = pd.read_csv(input_path)
        scored = predict(new_customers, model, scaler)
        output_path = input_path.replace(".csv", "_scored.csv")
        scored.to_csv(output_path, index=False)
        print(f"Scored {len(scored)} customers -> {output_path}")
        print(scored['risk_band'].value_counts())
    else:
        # example customer (a 52 year old inactive German customer with high balance)
        example = pd.DataFrame([{
            'credit_score': 600,
            'country': 'Germany',
            'gender': 'Female',
            'age': 52,
            'tenure': 3,
            'balance': 130000,
            'products_number': 1,
            'credit_card': 1,
            'active_member': 0,
            'estimated_salary': 90000
        }])
        result = predict(example, model, scaler)
        row = result.iloc[0]
        print("Example customer:")
        print(example.T.to_string(header=False))
        print("\nChurn probability:", row['churn_probability'])
        print("Risk band:", row['risk_band'])
        print("Will churn?", "YES - contact for retention" if row['predicted_churn'] == 1 else "No")
