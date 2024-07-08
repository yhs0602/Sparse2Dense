# Merge all the rows in the csv files

import os

import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    length_dir = "lengths"
    lengths = os.listdir(length_dir)
    all_data = []
    for length in tqdm(lengths):
        if not length.endswith(".gz"):
            continue
        data = pd.read_csv(os.path.join(length_dir, length), compression="gzip")
        all_data.append(data)
    all_data = pd.concat(all_data)
    all_data.to_csv("all_lengths.csv")
