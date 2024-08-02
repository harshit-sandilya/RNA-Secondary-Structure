import pandas as pd

df = pd.read_csv("./data/final/train.csv")
print(df.head())
df = df["sequence"].values
ranges = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in ranges:
    for j in df:
        if len(j) <= i:
            counts[ranges.index(i)] += 1
counts = [i - j for i, j in zip(counts, [0] + counts[:-1])]
print(counts)
ranges = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
for i in ranges:
    for j in df:
        if len(j) <= i:
            counts[ranges.index(i)] += 1
counts = [i - j for i, j in zip(counts, [0] + counts[:-1])]
print(counts)
