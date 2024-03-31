import datetime
from io import StringIO

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def visualize():
    # 로그 데이터 예시 (여기서는 문자열로부터 직접 생성하지만, 파일에서 읽을 수도 있음)
    data = """
time,type,event
1900-01-01 19:51:41.301433, start, Minecraft_env/onInitialize/Accept
1900-01-01 19:51:42.172467, start, read_response
1900-01-01 19:51:42.174127, end, Minecraft_env/onInitialize/Accept
1900-01-01 19:51:42.176395, start, Minecraft_env/onInitialize/readInitialEnvironment
1900-01-01 19:51:42.211516, end, Minecraft_env/onInitialize/readInitialEnvironment
1900-01-01 19:51:43.395444, start, Minecraft_env/onInitialize/ClientTick
1900-01-01 19:51:43.395531, start, Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick
1900-01-01 19:51:43.414041, end, Minecraft_env/onInitialize/ClientTick
1900-01-01 19:51:43.494428, start, Minecraft_env/onInitialize/ClientTick
1900-01-01 19:51:43.494518, start, Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick
1900-01-01 19:51:44.923876, end, Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick
1900-01-01 19:51:44.923944, end, Minecraft_env/onInitialize/ClientTick
1900-01-01 19:51:44.941409, start, Minecraft_env/onInitialize/ClientTick
1900-01-01 19:51:44.941484, start, Minecraft_env/onInitialize/ClientTick/EnvironmentInitializer/onClientTick
1900-01-01 19:51:49.872379, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.872606, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.890201, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.890250, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.892291, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.892355, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.893387, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.893408, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.893433, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.893483, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.894146, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.894168, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.894190, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.894221, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.894903, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.894925, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.894951, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.894991, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.895588, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.895607, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.895629, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.895658, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.896263, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.896283, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.896308, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.896336, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.897385, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.897403, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.897422, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.897448, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.898094, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.898114, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.898136, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.898163, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.898776, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.898792, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.898811, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.898836, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.899427, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.899443, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.899461, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.899486, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.900070, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.900087, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.900105, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.900138, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.900729, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.900746, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.900764, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.900790, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.901676, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.901697, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.901739, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.901770, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.902474, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.902496, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.902515, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.902538, end, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
1900-01-01 19:51:49.903438, start, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.903458, end, Minecraft_env/onInitialize/EndServerTick/NotifyClientSendObservation
1900-01-01 19:51:49.903485, start, Minecraft_env/onInitialize/StartServerTick/WaitClientAction
    """

    df = pd.read_csv(StringIO(data))

    # 'time' 열을 datetime으로 변환
    df["time"] = pd.to_datetime(df["time"])

    # 이벤트 시작과 종료를 추적하기 위한 딕셔너리
    event_durations = {}

    # 시작 시간과 종료 시간 계산
    for _, row in df.iterrows():
        event = row["event"]
        if row["type"].strip() == "start":
            event_durations[event] = [row["time"], None]
        elif row["type"].strip() == "end":
            if event in event_durations:
                event_durations[event][1] = row["time"]
    print(event_durations)
    # 시각화
    fig, ax = plt.subplots(figsize=(16, 6))

    # 이벤트 라벨과 그에 대응하는 막대(artist)를 저장하기 위한 딕셔너리
    artists = {}

    for i, (event, times) in enumerate(event_durations.items()):
        start, end = times
        # 이벤트 지속 시간을 계산 (초 단위)
        if start is None or end is None:
            continue
        if event.strip() == "Minecraft_env/onInitialize/Accept":
            continue
        duration: datetime = end - start  # .total_seconds()
        artist = ax.barh(
            event,
            duration.total_seconds() / (24 * 3600),
            left=mdates.date2num(start),
            height=0.4,
        )
        # 고유 이벤트에 대한 막대(artist)를 그리고, artists 딕셔너리에 저장
        # artist = ax.barh(event, duration, left=start, height=0.4)
        artists[event] = artist

    # 날짜 포맷터 설정
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    plt.xlabel("Time")
    plt.title("Event Durations")
    plt.legend([artists[event][0] for event in artists], [event for event in artists])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()
