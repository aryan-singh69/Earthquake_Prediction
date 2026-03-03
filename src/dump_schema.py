import pandas as pd
df = pd.read_csv("merge.csv", nrows=1, low_memory=False)
lines = []
lines.append("Columns: " + ", ".join(list(df.columns)))
lines.append("First Row: " + str(df.iloc[0].to_dict()))
with open("dump.txt", "w") as f:
    f.write("\n".join(lines))
