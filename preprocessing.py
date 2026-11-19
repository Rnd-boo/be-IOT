
import pandas as pd

# 1. Load dataset
df = pd.read_csv("sensor_data1.csv")

# 2. Convert timestamp
df["recorded_at"] = pd.to_datetime(
    df["recorded_at"], utc=True
)

# 3. Sort oldest to newest
df = df.sort_values("recorded_at").reset_index(drop=True)

# 4. Create next reading information
df["next_timestamp"] = df["recorded_at"].shift(-1)
df["target_power"] = df["power"].shift(-1)

# 5. Calculate interval
df["interval"] = (
    df["next_timestamp"] - df["recorded_at"]
)

# 6. Keep readings approximately 30 minutes apart
tolerance = pd.Timedelta(minutes=2)

valid = (
    (df["interval"] - pd.Timedelta(minutes=30)).abs()
    <= tolerance
)

data = df.loc[valid].copy()

# 7. Create time features (local Indonesian time)
local_time = data["recorded_at"].dt.tz_convert(
    "Asia/Jakarta"
)

import numpy as np

hour = (
    local_time.dt.hour
    + local_time.dt.minute / 60
)

data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
data["hour_cos"] = np.cos(2 * np.pi * hour / 24)

data["day_of_week"] = local_time.dt.dayofweek

# 8. Select ML features
features = [
    "voltage",
    "current",
    "power",
    "panel_temperature",
    "light_lux",
    "air_temperature",
    "humidity",
    "hour_sin",
    "hour_cos",
    "day_of_week"
]

X = data[features]
y = data["target_power"]

# 9. Export prepared dataset
prepared = X.copy()
prepared["target_power"] = y

prepared.to_csv("training_data.csv", index=False)

print("PREPROCESSING COMPLETE")
print("Original records:", len(df))
print("Valid training records:", len(prepared))
print("Missing feature values:")
print(X.isnull().sum())
print("\nPrepared dataset:")
print(prepared.head())
