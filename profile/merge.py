import pandas as pd

if __name__ == "__main__":
    # Log file path
    java_logfile1 = "fixed_java_log_filtered.csv"
    logfile2 = "py_log_1.csv"
    merged_logfile = "merged_log.csv"

    # Read each log file as a pandas DataFrame
    df1 = pd.read_csv(java_logfile1, header=None, names=["time", "type", "event"])
    df2 = pd.read_csv(logfile2, header=None, names=["time", "type", "event"])

    merged_df = pd.concat([df1, df2])

    # Convert column 'time' to datetime format
    merged_df["time"] = pd.to_datetime(merged_df["time"], format="%H:%M:%S.%f")

    merged_df = merged_df.sort_values(by="time")

    merged_df.to_csv(merged_logfile, index=False, header=True)

    print(f"Merged log file saved as {merged_logfile}.")
