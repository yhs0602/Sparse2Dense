import gzip
import os
from os.path import expanduser

import pandas as pd


def main():
    directory = expanduser("~/Downloads/representation_data_fixed/data")

    yaw_columns = []
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
            # select only position_yaw columns
            # concat horizntally
            yaw_columns.append(data["position_yaw"])
    yaw_data = pd.concat(yaw_columns, axis=1)
    yaw_data.to_csv(
        expanduser("~/Downloads/representation_data_fixed/yaw.csv"), index=False
    )


if __name__ == "__main__":
    main()
