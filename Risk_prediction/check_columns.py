
import pandas as pd

df = pd.read_pickle("clean_merged.pkl")

# Print ALL column names
for i, col in enumerate(df.columns, 1):
    print(f"{i:3}: {col}")

print("Total columns:", len(df.columns))
