import pandas as pd

df = pd.read_csv("data/transactions.csv")

print(df["TIMESTAMP"].head(20))
print("\nData type:", df["TIMESTAMP"].dtype)
print("Minimum:", df["TIMESTAMP"].min())
print("Maximum:", df["TIMESTAMP"].max())
print("Unique values:", df["TIMESTAMP"].nunique())