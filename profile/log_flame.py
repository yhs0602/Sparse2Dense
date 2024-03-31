import colorsys

import pandas as pd
import plotly.figure_factory as ff


def generate_colors(N):
    colors = []
    for i in range(N):
        hue = i / N
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append("#" + "".join([format(int(x * 255), "02x") for x in rgb]))
    return colors


def visualize():
    # 로그 데이터 불러오기
    df = pd.read_csv("merged_log.csv")
    print("Read log data")
    # 'time' 열을 datetime으로 변환
    df["time"] = pd.to_datetime(df["time"])

    # Gantt 차트에 사용할 데이터 준비
    gantt_data = []
    processed = 0
    # Key: event name, Value: last start event buffer
    last_buffers = {}
    sliced_df = df.iloc[:100000]

    start_df = (
        sliced_df[sliced_df["type"] == " start"]
        .rename(columns={"time": "start_time"})
        .drop("type", axis=1)
    )
    end_df = (
        sliced_df[sliced_df["type"] == " end"]
        .rename(columns={"time": "end_time"})
        .drop("type", axis=1)
    )

    # 시작과 종료 이벤트 결합
    merged_df = pd.merge(start_df, end_df, on="event")

    # 각 이벤트의 지속 시간 계산
    merged_df["duration"] = (
        merged_df["end_time"] - merged_df["start_time"]
    ).dt.total_seconds()

    # 이벤트별 평균 지속 시간 계산
    average_durations = merged_df.groupby("event")["duration"].mean().reset_index()

    # 결과 출력
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    average_durations_sorted = average_durations.sort_values(
        by="duration", ascending=False
    ).reset_index(drop=True)
    print(average_durations_sorted)
    for _, row in sliced_df.iterrows():
        if row["type"].strip() == "start":
            last_buffers[row["event"]] = [row, None]
        elif row["type"].strip() == "end":
            last_item = last_buffers.get(row["event"], None)
            if last_item is None:
                print(f"No start event found for {row['event']}; i={processed}")
                continue
            gantt_data.append(
                dict(
                    Task=row["event"],
                    Start=last_item[0]["time"],
                    Finish=row["time"],
                    Resource=row["event"],
                )
            )
            last_buffers.pop(row["event"])
        processed += 1
        if processed % 100000 == 0:
            print(f"Processed {processed} rows")
    print("Creating Gantt data")
    unique_groups = sliced_df["event"].nunique()
    print(f"Unique colors: {unique_groups}")
    colors_hex = generate_colors(unique_groups)
    # Gantt 차트 생성
    fig = ff.create_gantt(
        gantt_data,
        index_col="Resource",
        title="Event Durations Gantt Chart",
        show_colorbar=True,
        group_tasks=True,
        colors=colors_hex,
    )

    # 차트 보기
    fig.show()


if __name__ == "__main__":
    visualize()
