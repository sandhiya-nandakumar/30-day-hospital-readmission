import sqlite3
import pandas as pd
import os

db_file = "predictions.db"
output_file = "powerbi_predictions.csv"

conn = sqlite3.connect(db_file)

df = pd.read_sql_query(
    "SELECT * FROM predictions",
    conn
)

conn.close()

df.to_csv(output_file, index=False)

print("Prediction history exported successfully!")
print(f"File: {os.path.abspath(output_file)}")
print(f"Rows: {len(df)}")
print("\nColumns:")
print(df.columns.tolist())