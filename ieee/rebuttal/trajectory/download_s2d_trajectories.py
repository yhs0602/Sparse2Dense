import os
from datetime import datetime

import pandas as pd
import wandb
from tqdm import tqdm
from wandb.apis.public import Run, Runs

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_canonical_directory = os.path.realpath(current_directory)


def download_wandb_file(run: Run, download_dir: str):
    run_id = run.id
    run_group = run.group

    group_dir = os.path.join(download_dir, run_group)
    os.makedirs(group_dir, exist_ok=True)

    file_path = os.path.join(group_dir, f"{run_id}.csv.gz")
    print(f"Downloading {run_id} to {file_path}")

    if not os.path.exists(file_path):
        scan_history = run.scan_history(
            keys=["global_step", "episode", "episode/positions"]
        )
        data = [
            [
                row.get(column)
                for column in ["global_step", "episode", "episode/positions"]
            ]
            for row in scan_history
        ]
        df = pd.DataFrame(data, columns=["global_step", "episode", "episode/positions"])
        df.to_csv(file_path, compression="gzip")
        # history = run.history(keys=["episode", "episode/positions"], samples=500)
        # history.to_csv(file_path)
    else:
        print(f"File {file_path} already exists")
        return


def get_run(seed, transition_timing, timing) -> str:
    sparse_base_3M = {
        0: "nk51vw9k",
        42: "9gnwujc3",
        9876: "axg2oaun",
        7777: "f82umoxi",
        2024: "7mek9rlt",
        1234: "yijbu94z",  # "elimafnz": kube4
    }

    sparse_base_5M = {
        0: "uwf6ucxe",
        42: "ni6k4i1i",
        9876: "6ydi4iup",
        7777: "535ibmrw",
        2024: "pvsbvpsw",
        1234: "u1nytrvk",
    }

    # ysjw4040	6pm4fe1v	15h89j7q	4r70xs2o	wye1y1c8	6k0xnhao
    dense_1M = {
        0: "ysjw4040",
        42: "6pm4fe1v",
        9876: "15h89j7q",
        7777: "4r70xs2o",
        2024: "wye1y1c8",
        1234: "6k0xnhao",
    }

    # 4cz52gwc	x5ixjak6	jpce3981	epko7wk3	eelzon7x	evn1216o
    dense_2M = {
        0: "4cz52gwc",
        42: "x5ixjak6",
        9876: "jpce3981",
        7777: "epko7wk3",
        2024: "eelzon7x",
        1234: "evn1216o",
    }

    # 3b0p40ay	(kxbvd0pn, pjks8til, xjdz4427)	owj5ut16	vpkpn9z2	760l11ru	d9j891eg, mpo99h1p, rwfzeetw, sexn6r3b
    dense_3M = {
        0: "3b0p40ay",
        42: "kxbvd0pn",  # kxbvd0pn, pjks8til, xjdz4427
        9876: "owj5ut16",
        7777: "vpkpn9z2",
        2024: "760l11ru",
        1234: "d9j891eg",  # d9j891eg, mpo99h1p, rwfzeetw, sexn6r3b
    }

    # s0o6usad	l2h2e35o	sap2y5y1	sg55x5da	spubrl80	hheqgwfk
    dense_4M = {
        0: "s0o6usad",
        42: "l2h2e35o",
        9876: "sap2y5y1",
        7777: "sg55x5da",
        2024: "spubrl80",
        1234: "hheqgwfk",
    }

    # 2zsdgypz	sia1cf45	s2v6wzk3	13nc4v8m	ns8pdmi0	vjypuoac
    dense_5M = {
        0: "2zsdgypz",
        42: "sia1cf45",
        9876: "s2v6wzk3",
        7777: "13nc4v8m",
        2024: "ns8pdmi0",
        1234: "vjypuoac",
    }
    # 1. check if it is sparse or dense timing
    # 2. select the base run if sparse and select the dense run if after timing
    # 3. select by the seed

    s2d_1M = [
        sparse_base_3M,  # 0
        sparse_base_3M,  # 100만
        dense_1M,  # 200만
        dense_1M,  # 300만
        dense_1M,  # 400만
        dense_1M,  # 500만
        dense_1M,  # 600만
        dense_1M,  # 700만
        dense_1M,  # 800만
        dense_1M,  # 900만
        dense_1M,  # 1000만
    ]

    s2d_2M = [
        sparse_base_3M,  # 0
        sparse_base_3M,  # 100만
        sparse_base_3M,  # 200만
        dense_2M,  # 300만
        dense_2M,  # 400만
        dense_2M,  # 500만
        dense_2M,  # 600만
        dense_2M,  # 700만
        dense_2M,  # 800만
        dense_2M,  # 900만
        dense_2M,  # 1000만
    ]

    s2d_3M = [
        sparse_base_3M,  # 0
        sparse_base_3M,  # 100만
        sparse_base_3M,  # 200만
        sparse_base_3M,  # 300만
        dense_3M,  # 400만
        dense_3M,  # 500만
        dense_3M,  # 600만
        dense_3M,  # 700만
        dense_3M,  # 800만
        dense_3M,  # 900만
        dense_3M,  # 1000만
    ]

    s2d_4M = [
        sparse_base_3M,  # 0
        sparse_base_3M,  # 100만
        sparse_base_3M,  # 200만
        sparse_base_3M,  # 300만
        sparse_base_5M,  # 400만
        dense_4M,  # 500만
        dense_4M,  # 600만
        dense_4M,  # 700만
        dense_4M,  # 800만
        dense_4M,  # 900만
        dense_4M,  # 1000만
    ]

    s2d_5M = [
        sparse_base_3M,  # 0
        sparse_base_3M,  # 100만
        sparse_base_3M,  # 200만
        sparse_base_3M,  # 300만
        sparse_base_5M,  # 400만
        sparse_base_5M,  # 500만
        dense_5M,  # 600만
        dense_5M,  # 700만
        dense_5M,  # 800만
        dense_5M,  # 900만
        dense_5M,  # 1000만
    ]

    return {
        1: s2d_1M,
        2: s2d_2M,
        3: s2d_3M,
        4: s2d_4M,
        5: s2d_5M,
    }[transition_timing][
        timing
    ][seed]


def main():
    all_runs = set()
    for seed in [0, 42, 9876, 7777, 2024, 1234]:
        for transition_timing in [1, 2, 3, 4, 5]:
            for timing in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                all_runs.add(get_run(seed, transition_timing, timing))
    print(all_runs)
    api = wandb.Api(timeout=120)
    WANDB_ENTITY = "jourhyang123"
    WANDB_PROJECT = "Sparse2Dense-room_experiments"
    for run_id in tqdm(all_runs):
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        download_wandb_file(run, "./all_run_trajectories-s2d")


if __name__ == "__main__":
    main()
