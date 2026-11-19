
import pandas as pd

df = pd.read_csv("sensor_data1.csv")

df["recorded_at"] = pd.to_datetime(
    df["recorded_at"], utc=True
)

df = df.sort_values("recorded_at")

# Calculate the difference between consecutive readings
intervals = df["recorded_at"].diff()

print("INTERVAL DISTRIBUTION:")
print(intervals.value_counts())

print("\nTOTAL RECORDS:", len(df))

print("\nEXPECTED INTERVAL: 30 minutes")
print(
    "ACTUAL 30-MINUTE INTERVALS:",
    (intervals == pd.Timedelta(minutes=30)).sum()
)

print(
    "MISSING INTERVALS:",
    (intervals > pd.Timedelta(minutes=30)).sum()
)

print("\nDATE RANGE:")
print(df["recorded_at"].min(), "to", df["recorded_at"].max())
