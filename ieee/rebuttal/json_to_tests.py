import json
from typing import Dict, Tuple
import numpy as np
from scipy.stats import f_oneway


def t_test():
    pass


def anova(groups: Dict[str, Tuple[float, float]], n=7):
    # 그룹별 mean/std
    groups = {
        "S2D(C_1)": (138.71, 3.71),
        "S2D(C_2)": (63.40, 160.55),
        "S2D(C_3)": (168.88, 23.66),
        "OnlySparse": (142.50, 4.25),
        "OnlyDense": (139.68, 14.90),
        "D2S(C_1)": (140.75, 7.46),
        "D2S(C_2)": (130.63, 19.69),
    }
    # 가상 샘플 생성
    samples = []
    for mean, std in groups.values():
        s = np.random.normal(loc=mean, scale=std, size=n)
        samples.append(s)

    # ANOVA
    anova_result = f_oneway(*samples)
    return anova_result


def main():
    with open("table.json", "r") as f:
        data = json.load(f)
    anova_results = {}
    for row in data:
        print("===" * 10)
        print("Metric: ", row["TaskMetric"]["value"])
        print("==" * 10)
        print("S2D(C_1): ", row["S2D(C_1)"]["value"], "±", row["S2D(C_1)"]["stdv"])
        print("S2D(C_2): ", row["S2D(C_2)"]["value"], "±", row["S2D(C_2)"]["stdv"])
        print("S2D(C_3): ", row["S2D(C_3)"]["value"], "±", row["S2D(C_3)"]["stdv"])
        print(
            "Only Sparse: ", row["OnlySparse"]["value"], "±", row["OnlySparse"]["stdv"]
        )
        print("Only Dense: ", row["OnlyDense"]["value"], "±", row["OnlyDense"]["stdv"])
        print("D2S(C_1): ", row["D2S(C_1)"]["value"], "±", row["D2S(C_1)"]["stdv"])
        print("D2S(C_2): ", row["D2S(C_2)"]["value"], "±", row["D2S(C_2)"]["stdv"])

        groups = {
            "S2D(C_1)": (row["S2D(C_1)"]["value"], row["S2D(C_1)"]["stdv"]),
            "S2D(C_2)": (row["S2D(C_2)"]["value"], row["S2D(C_2)"]["stdv"]),
            "S2D(C_3)": (row["S2D(C_3)"]["value"], row["S2D(C_3)"]["stdv"]),
            "OnlySparse": (row["OnlySparse"]["value"], row["OnlySparse"]["stdv"]),
            "OnlyDense": (row["OnlyDense"]["value"], row["OnlyDense"]["stdv"]),
            "D2S(C_1)": (row["D2S(C_1)"]["value"], row["D2S(C_1)"]["stdv"]),
            "D2S(C_2)": (row["D2S(C_2)"]["value"], row["D2S(C_2)"]["stdv"]),
        }
        anova_result = anova(groups)
        print(
            f"ANOVA result: statistic={anova_result.statistic:.2f}, p-value={anova_result.pvalue:.2f}"
        )
        anova_results[row["TaskMetric"]["value"]] = {
            "statistic": anova_result.statistic,
            "p-value": anova_result.pvalue,
        }
    print(json.dumps(anova_results, indent=4))
    with open("anova_results.json", "w") as f:
        json.dump(anova_results, f, indent=4)


if __name__ == "__main__":
    main()
