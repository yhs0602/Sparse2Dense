import gzip
import os
from os.path import expanduser

import pandas as pd


def main():
    directory = expanduser("~/Downloads/representation_data_fixed/data")

    for dir in os.listdir(directory):
        if not dir.endswith("json.gz"):
            continue
        path = os.path.join(directory, dir)
        print(f"Reading {path}")
        with gzip.open(path, "rt") as f:
            data = pd.read_json(f)
            print(data.head())
            print(data.columns)
            print(data.shape)


if __name__ == "__main__":
    main()
