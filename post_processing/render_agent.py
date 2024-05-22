import matplotlib.pyplot as plt
import time


def draw_maze_and_agent(maze, position, direction):
    maze_x_size = len(maze[0])
    maze_z_size = len(maze)

    fig, ax = plt.subplots()
    ax.set_xlim([0, maze_x_size])
    ax.set_ylim([0, maze_z_size])

    # 미로 그리기
    for z in range(maze_z_size):
        for x in range(maze_x_size):
            if maze[z][x] == "o":
                rect = plt.Rectangle((x, maze_z_size - z - 1), 1, 1, color="black")
                ax.add_patch(rect)

    # 에이전트 위치와 방향 계산
    x, z = position
    z = maze_z_size - z - 1  # y축 반전

    # 에이전트의 방향을 기반으로 삼각형의 꼭짓점 계산
    triangle_size = 0.3
    if direction == "N":  # 북
        vertices = [
            (x, z + triangle_size),
            (x - triangle_size, z - triangle_size),
            (x + triangle_size, z - triangle_size),
        ]
    elif direction == "S":  # 남
        vertices = [
            (x, z - triangle_size),
            (x - triangle_size, z + triangle_size),
            (x + triangle_size, z + triangle_size),
        ]
    elif direction == "E":  # 동
        vertices = [
            (x + triangle_size, z),
            (x - triangle_size, z + triangle_size),
            (x - triangle_size, z - triangle_size),
        ]
    elif direction == "W":  # 서
        vertices = [
            (x - triangle_size, z),
            (x + triangle_size, z + triangle_size),
            (x + triangle_size, z - triangle_size),
        ]

    # 삼각형 그리기
    triangle = plt.Polygon(vertices, closed=True, color="red")
    ax.add_patch(triangle)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# 예시 미로 데이터
maze = [
    "oooxxxxooo",
    "oxoxxxxoxo",
    "oxooooooxo",
    "oxxxxxxxxo",
    "oxooooooxo",
    "oxoxxxxoxo",
    "oooxxxxooo",
]

if __name__ == "__main__":
    # 에이전트 위치 및 방향
    position = (5, 3)  # (x, z)
    direction = "N"  # 방향: North (북)
    start_time = time.time_ns()
    draw_maze_and_agent(maze, position, direction)
    end_time = time.time_ns()
    print(f"Time: {end_time - start_time} ns, fps={1e9 / (end_time - start_time)}")
