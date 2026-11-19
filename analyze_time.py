
import pandas as pd

df = pd.read_csv("training_data.csv")

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

print("POWER BY HOUR")

print(
    df.groupby("hour", observed=False)["power"]
    .agg(["count", "mean", "max"])
    .round(2)
)

print("\nPOWER CATEGORY BY HOUR")

print(
    pd.crosstab(
        df["hour"],
        df["power_category"]
    )
)
