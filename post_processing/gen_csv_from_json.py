import pandas as pd

data = {
    "v30-crossw2-sparse-2": {
        "mean_length": 1471.0238037109375,
        "mean_reward": 0.8528976099831718,
        "mean_eval_episodes": 318.85714285714283,
        "mean_eval_left_success_count": 101.57142857142857,
        "mean_eval_middle_success_count": 102.57142857142857,
        "mean_eval_right_success_count": 101.28571428571429,
        "mean_eval_total_success_count": 305.42857142857144,
    },
    "v30-crossw2-dense-0": {
        "mean_length": 898.04443359375,
        "mean_reward": 0.9136844674746195,
        "mean_eval_episodes": 330.6666666666667,
        "mean_eval_left_success_count": 106.66666666666667,
        "mean_eval_middle_success_count": 106.33333333333333,
        "mean_eval_right_success_count": 106.0,
        "mean_eval_total_success_count": 319.0,
    },
    "v30-crossw2-dense-2": {
        "mean_length": 1221.6778055826824,
        "mean_reward": 0.8820211191972097,
        "mean_eval_episodes": 315.1666666666667,
        "mean_eval_left_success_count": 100.0,
        "mean_eval_middle_success_count": 101.66666666666667,
        "mean_eval_right_success_count": 100.0,
        "mean_eval_total_success_count": 301.6666666666667,
    },
    "v30-crossw2-transition-3000000-1": {
        "mean_length": 1214.7388712565105,
        "mean_reward": 0.87852610150973,
        "mean_eval_episodes": 423.6666666666667,
        "mean_eval_left_success_count": 135.83333333333334,
        "mean_eval_middle_success_count": 136.66666666666666,
        "mean_eval_right_success_count": 135.33333333333334,
        "mean_eval_total_success_count": 407.8333333333333,
    },
    "v30-crossw2-transition-3000000-2": {
        "mean_length": 2131.8943990071616,
        "mean_reward": 0.7868105471134186,
        "mean_eval_episodes": 310.0,
        "mean_eval_left_success_count": 99.5,
        "mean_eval_middle_success_count": 99.83333333333333,
        "mean_eval_right_success_count": 99.83333333333333,
        "mean_eval_total_success_count": 299.1666666666667,
    },
    "v30-crossw2-transition-3000000-0": {
        "mean_length": 1356.7611083984375,
        "mean_reward": 0.8643238743146261,
        "mean_eval_episodes": 434.0,
        "mean_eval_left_success_count": 137.0,
        "mean_eval_middle_success_count": 139.5,
        "mean_eval_right_success_count": 136.5,
        "mean_eval_total_success_count": 413.0,
    },
    "v30-crossw2-sparse-0": {
        "mean_length": 1295.8571602957588,
        "mean_reward": 0.8704142825944083,
        "mean_eval_episodes": 363.14285714285717,
        "mean_eval_left_success_count": 115.85714285714286,
        "mean_eval_middle_success_count": 116.85714285714286,
        "mean_eval_right_success_count": 115.28571428571429,
        "mean_eval_total_success_count": 348.0,
    },
    "v30-crossw2-dense-1": {
        "mean_length": 983.9277648925781,
        "mean_reward": 0.9055627882480621,
        "mean_eval_episodes": 392.6666666666667,
        "mean_eval_left_success_count": 126.33333333333333,
        "mean_eval_middle_success_count": 126.5,
        "mean_eval_right_success_count": 126.66666666666667,
        "mean_eval_total_success_count": 379.5,
    },
    "v30-crossw2-sparse-1": {
        "mean_length": 1884.685756138393,
        "mean_reward": 0.8067695157868522,
        "mean_eval_episodes": 318.85714285714283,
        "mean_eval_left_success_count": 102.71428571428571,
        "mean_eval_middle_success_count": 102.42857142857143,
        "mean_eval_right_success_count": 102.57142857142857,
        "mean_eval_total_success_count": 307.7142857142857,
    },
}

if __name__ == "__main__":
    df = pd.DataFrame.from_dict(data, orient="index")

    # Save as CSV file
    csv_file = "./experiment_results.csv"
    df.to_csv(csv_file)

    print(f"CSV file created at {csv_file}")
