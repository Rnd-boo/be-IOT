
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load prepared dataset
df = pd.read_csv("training_data.csv")

# 2. Separate features and target
X = df.drop(columns=["target_power"])
y = df["target_power"]

# 3. Chronological train/test split
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Create XGBoost model
model = XGBRegressor(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

# 5. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")

# 6. Predict testing data
predictions = model.predict(X_test)

# 7. Evaluate model
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION")
print("MAE:", round(mae, 3), "W")
print("RMSE:", round(rmse, 3), "W")
print("R2:", round(r2, 3))

# 8. Compare against a simple baseline
baseline_predictions = X_test["power"]

baseline_mae = mean_absolute_error(
    y_test, baseline_predictions
)

print("\nBASELINE COMPARISON")
print("Baseline MAE:", round(baseline_mae, 3), "W")
print("XGBoost MAE:", round(mae, 3), "W")

# 9. Save model
model.save_model("solar_xgboost.json")

# 10. Save predictions
results = pd.DataFrame({
    "actual_power": y_test.values,
    "predicted_power": predictions
})

results.to_csv("predictions.csv", index=False)

print("\nModel saved: solar_xgboost.json")
print("Predictions saved: predictions.csv")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.plot(
    results["actual_power"].values,
    label="Actual Power"
)

plt.plot(
    results["predicted_power"].values,
    label="Predicted Power"
)

plt.xlabel("Testing Records")
plt.ylabel("Power (W)")
plt.title("Actual vs Predicted Solar Power")

plt.legend()
plt.grid(True)
plt.show()