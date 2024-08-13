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
    # Importing log data
    df = pd.read_csv("merged_log.csv")
    print("Read log data")
    # Convert 'time' column to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Preparing data for Gantt charts
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

    # Combining start and end events
    merged_df = pd.merge(start_df, end_df, on="event")

    # Calculate the duration of each event
    merged_df["duration"] = (
        merged_df["end_time"] - merged_df["start_time"]
    ).dt.total_seconds()

    # Calculate average duration per event
    average_durations = merged_df.groupby("event")["duration"].mean().reset_index()

    # Output the result
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
    # Create a Gantt chart
    fig = ff.create_gantt(
        gantt_data,
        index_col="Resource",
        title="Event Durations Gantt Chart",
        show_colorbar=True,
        group_tasks=True,
        colors=colors_hex,
    )

    fig.show()


if __name__ == "__main__":
    visualize()
