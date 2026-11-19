
import pandas as pd

df = pd.read_csv("training_data.csv")

print("POWER STATISTICS")
print(df["power"].describe())

print("\nPOWER DISTRIBUTION")

df["power_category"] = pd.cut(
    df["power"],
    bins=[-1, 1, 5, 15, 25, 100],
    labels=[
        "Very Low",
        "Low",
        "Medium",
        "High",
        "Very High"
    ]
)

print(df["power_category"].value_counts())
