import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/sales.csv"
MODEL_PATH = "model/model.joblib"


def main():
    # 1. Load the data
    df = pd.read_csv(DATA_PATH)

    # 2. Split into input (X) and target (y)
    X = df[["Temperature"]]  # double brackets keep it a table, which sklearn expects
    y = df["Revenue"]

    # 3. Hold out 20% of the rows to test the model on unseen data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Evaluate on the test set
    y_pred = model.predict(X_test)
    print(f"R2 score (test): {r2_score(y_test, y_pred):.4f}")
    print(f"MAE (test):      {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Slope:           {model.coef_[0]:.2f} revenue per degree")
    print(f"Intercept:       {model.intercept_:.2f}")

    # 6. Save the trained model so the API can load it later
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()