import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

url = f"{SUPABASE_URL}/rest/v1/sensor_readings"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}"
}

params = {
    "select": "*",
    "order": "id",
}

response = requests.get(url, headers=headers, params=params)

response.raise_for_status()

data = response.json()

# Display raw data
df = pd.DataFrame(data)

# Export to CSV
df.to_csv("sensor_data1.csv", index=False)

print("Data successfully exported!")
print(f"Total records: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print("File: sensor_data1.csv")