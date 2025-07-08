import json
import re

latex = """
Task Metric & S2D(   {C}_1  )    & S2D(   {C}_2  )     &   {S2D}(   {C}_3  )   & Only Sparse & Only Dense & D2S(   {C}_1  ) & D2S(   {C}_2  ) & D2S(   {C}_3  )  
Lunar Lander Perf. & 138.71\stdv{3.71}          & 63.40\stdv{160.55}   &   {168.88}\stdv{23.66}   & 142.50\stdv{4.25}          & 139.68\stdv{14.90}          &    140.75\stdv{7.46}     & 130.63\stdv{19.69}  &   142.37\stdv{15.62}
Lunar Lander Sharp. & 27.06\stdv{36.31}  & 1231.93\stdv{2424.61}     &   {7.46}\stdv{3.37} & 8.97\stdv{2.83}  & 8.71\stdv{4.43}   &   8.95\stdv{2.89}  & 8.99\stdv{2.97}& 11.32\stdv{3.72}
CartPole Perf.    & 3.18\stdv{4.00} &   {14.61}\stdv{10.96}   & 5.29\stdv{7.47}  & 0.14\stdv{0.25}  & 3.88\stdv{4.63}  & 1.55\stdv{0.29} & 0.38\stdv{0.07} & 0.97\stdv{0.19}
CartPole Sharp. & 0.12\stdv{0.24}          &   {0.01\stdv{0.15}}          & 0.01\stdv{0.24}          & 0.08\stdv{0.57}        & 0.19\stdv{0.03}          &  0.16\stdv{0.09} & 0.05\stdv{0.21} & 0.02\stdv{0.17} 
UR5 Perf.  & 65.54\stdv{10.86}  & 65.69\stdv{17.32}      &   {94.15}\stdv{4.28}      & 0.00\stdv{0.00}          & 64.23\stdv{13.03}          & 0.00\stdv{0.00}   & 0.00\stdv{0.00} & 0.00\stdv{0.00} 
UR5 Sharp. & 0.67\stdv{0.01}          & 0.62\stdv{0.11}          &   {0.61\stdv{0.04}}          & 0.09\stdv{0.52}          & 0.67\stdv{0.01}          & 0.52\stdv{0.24}  & 0.56\stdv{0.28} & 0.47\stdv{0.20} 
Cross Maze 0  Perf.    & 75.19\stdv{0.06}  &   {81.90}\stdv{0.06}  & 75.49\stdv{0.06}  & 64.94\stdv{0.03}  & 67.32\stdv{0.08}  & 60.57\stdv{0.03}  & 62.88\stdv{0.03}  & 63.96\stdv{0.03} 
Cross Maze 1  Perf.    & 69.65\stdv{0.07}  &   {77.16}\stdv{0.05}  & 75.39\stdv{0.06}  & 57.82\stdv{0.03}  & 69.46\stdv{0.07}  & 55.66\stdv{0.04}  & 57.31\stdv{0.05}  & 51.95\stdv{0.05} 
Cross Maze 2  Perf.    &   {63.60}\stdv{0.05}  & 62.86\stdv{0.03}  & 57.60\stdv{0.06}  & 57.26\stdv{0.03}  & 57.54\stdv{0.07}  & 55.15\stdv{0.05}  & 57.25\stdv{0.04}  & 54.47\stdv{0.04} 
playroom maze Perf.   & 22.95\stdv{0.03}  & 21.94\stdv{0.02}  &   {25.78}\stdv{0.03}  & 17.50\stdv{0.02}  & 18.91\stdv{0.02}  & 16.06\stdv{0.01}  & 17.22\stdv{0.01}  & 17.59\stdv{0.01} 
"""


def main():
    rows = latex.splitlines()
    data = []
    parsed_rows = []
    titles = []

    for row in rows:
        parsed_cells = []
        parsed_row = {}
        row = row.strip()
        if not row:
            continue
        cells = row.split("&")
        for idx, cell in enumerate(cells):
            cell = cell.strip()
            if not cell:
                continue
            parsed_cells.append(cell)
            if titles:
                parsed_row[titles[idx]] = cell
        if not titles:
            parsed_cells = [
                s.replace(" ", "").replace("", "").replace("{", "").replace("}", "")
                for s in parsed_cells
            ]
            titles = parsed_cells
            continue
        parsed_rows.append(parsed_row)
        print(len(parsed_cells))
        print(parsed_cells)
    print(titles)
    print(parsed_rows)

    for parsed_row in parsed_rows:
        for title, value in parsed_row.items():
            if "\\stdv" in value:
                real_value = (
                    value.split("\\stdv")[0]
                    .strip()
                    .replace("\\", "")
                    .replace("{", "")
                    .replace("}", "")
                )
                stdv = (
                    value.split("\\stdv")[1].strip().replace("}", "").replace("{", "")
                )
                parsed_row[title] = {"value": real_value, "stdv": stdv}
            else:
                parsed_row[title] = {"value": value.strip(), "stdv": None}
    print(json.dumps(parsed_rows, indent=4))
    with open("table.json", "w") as f:
        json.dump(parsed_rows, f, indent=4)

        # # LaTeX stdv{} 치환
        # row = re.sub(r'stdv\{(.*?)\}', r'±\1', row)
        # # & 로 분할
        # cols = [col.strip() for col in row.split('&')]
        # data.append(cols)

    # import pandas as pd
    # df = pd.DataFrame(data)
    # df.to_csv("table.csv", index=False)


if __name__ == "__main__":
    main()
