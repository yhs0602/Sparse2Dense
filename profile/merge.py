import pandas as pd

if __name__ == "__main__":
    # 로그 파일 경로
    java_logfile1 = "fixed_java_log_filtered.csv"
    logfile2 = "py_log_1.csv"
    merged_logfile = "merged_log.csv"

    # 각 로그 파일을 pandas DataFrame으로 읽기
    df1 = pd.read_csv(java_logfile1, header=None, names=["time", "type", "event"])
    df2 = pd.read_csv(logfile2, header=None, names=["time", "type", "event"])

    # 두 DataFrame 합치기
    merged_df = pd.concat([df1, df2])

    # 'time' 열을 datetime 형식으로 변환
    merged_df["time"] = pd.to_datetime(merged_df["time"], format="%H:%M:%S.%f")

    # 시간 순으로 정렬
    merged_df = merged_df.sort_values(by="time")

    # 결과를 파일로 저장
    merged_df.to_csv(merged_logfile, index=False, header=True)

    print(f"Merged log file saved as {merged_logfile}.")
