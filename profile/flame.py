import plotly.figure_factory as ff

if __name__ == "__main__":
    # 예제 데이터: 각 이벤트의 시작 시간, 종료 시간, 이름을 포함
    df = [
        dict(
            Task="Minecraft_env/onInitialize/Accept",
            Start="2023-04-01 19:51:41",
            Finish="2023-04-01 19:51:42",
            Resource="Accept",
        ),
        dict(
            Task="read_response",
            Start="2023-04-01 19:51:42",
            Finish="2023-04-01 19:51:43",
            Resource="Response",
        ),
        dict(
            Task="Minecraft_env/onInitialize/readInitialEnvironment",
            Start="2023-04-01 19:51:42",
            Finish="2023-04-01 19:51:43",
            Resource="Environment",
        ),
        dict(
            Task="Minecraft_env/onInitialize/readInitialEnvironment",
            Start="2023-04-01 19:51:44",
            Finish="2023-04-01 19:51:46",
            Resource="Environment",
        ),
    ]

    # Gantt 차트 생성
    fig = ff.create_gantt(
        df,
        title="Method Profiling Gantt Chart",
        index_col="Resource",
        show_colorbar=True,
        group_tasks=True,
    )

    # 차트 보기
    fig.show()
