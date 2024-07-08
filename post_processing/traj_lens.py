# aggregate trajectory length for each runs in the trajectory and save it as a csv file
import json
import os

import pandas as pd


# csv cache paths: cache/run_id.csv.gz # btlv4am1.csv.gz
def main():
    cache_dir = "cache"
    output_dir = "lengths"
    # enumerate all runs in the directory
    runs = os.listdir(cache_dir)
    # iterate over each run
    for run in runs:
        # read the csv file
        data = pd.read_csv(os.path.join(cache_dir, run), compression="gzip")
        # aggregate the trajectory length
        data["episode/length"] = data["episode/positions"].apply(
            lambda x: len(json.loads(x))
        )
        filtered_data = data[
            (data["episode/length"] >= 2000) & (data["episode/length"] <= 3000)
        ]
        # save the aggregated data
        filtered_data.to_csv(os.path.join(output_dir, run), compression="gzip")


if __name__ == "__main__":
    main()
