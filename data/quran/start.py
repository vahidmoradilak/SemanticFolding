import pandas as pd

# JSONL version
df = pd.read_json("multilingual_quran.jsonl", lines=True)
print(df.head())

# Parquet version
df_parquet = pd.read_parquet("multilingual_quran.parquet")
# print(df_parquet.head())
print(df.head(1))