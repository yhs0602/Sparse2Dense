import os

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    runs = os.listdir("lengths")

    dfs = []
    for file in tqdm(runs):
        if file.endswith(".csv.gz"):
            df = pd.read_csv(f"lengths/{file}")
            # Find rows with _step=3111
            df = df[df["_step"] == 3111]
            # check if the df is empty
            if df.empty:
                continue
            else:
                print("name is ", file)
            dfs.append(df)
    all_df = pd.concat(dfs)
    all_df.to_csv("all_runs.csv.gz", compression="gzip")


# Answer: su80k2nq.csv.gz's 22 ; 3111
# Goal = [8.741072837046506, 2, 17.98566927436925]
# Length = 2647
