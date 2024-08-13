import matplotlib.pyplot as plt
import time


def draw_maze_and_agent(maze, position, direction):
    maze_x_size = len(maze[0])
    maze_z_size = len(maze)

    fig, ax = plt.subplots()
    ax.set_xlim([0, maze_x_size])
    ax.set_ylim([0, maze_z_size])

    # Maze Draw
    for z in range(maze_z_size):
        for x in range(maze_x_size):
            if maze[z][x] == "o":
                rect = plt.Rectangle((x, maze_z_size - z - 1), 1, 1, color="black")
                ax.add_patch(rect)

    # Calculating agent location and orientation
    x, z = position
    z = maze_z_size - z - 1  # Invert y-axis

    # Calculate the vertices of a triangle based on the Agent's orientation
    triangle_size = 0.3
    if direction == "N":  # North
        vertices = [
            (x, z + triangle_size),
            (x - triangle_size, z - triangle_size),
            (x + triangle_size, z - triangle_size),
        ]
    elif direction == "S":  # South
        vertices = [
            (x, z - triangle_size),
            (x - triangle_size, z + triangle_size),
            (x + triangle_size, z + triangle_size),
        ]
    elif direction == "E":  # East
        vertices = [
            (x + triangle_size, z),
            (x - triangle_size, z + triangle_size),
            (x - triangle_size, z - triangle_size),
        ]
    elif direction == "W":  # West
        vertices = [
            (x - triangle_size, z),
            (x + triangle_size, z + triangle_size),
            (x + triangle_size, z - triangle_size),
        ]

    # Triangle Draw
    triangle = plt.Polygon(vertices, closed=True, color="red")
    ax.add_patch(triangle)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


# Example Maze Data
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
    # Agent location and orientation
    position = (5, 3)  # (x, z)
    direction = "N"  # North
    start_time = time.time_ns()
    draw_maze_and_agent(maze, position, direction)
    end_time = time.time_ns()
    print(f"Time: {end_time - start_time} ns, fps={1e9 / (end_time - start_time)}")
