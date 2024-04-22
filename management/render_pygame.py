import time

import pygame


def create_maze_image_pygame(maze, cell_size=40):
    pygame.init()
    start_time = time.time_ns()
    width = len(maze[0]) * cell_size
    height = len(maze) * cell_size
    surface = pygame.Surface((width, height))

    # 미로 그리기
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            color = (0, 0, 0) if cell == "o" else (255, 255, 255)
            pygame.draw.rect(
                surface, color, (x * cell_size, y * cell_size, cell_size, cell_size)
            )
    end_time = time.time_ns()
    print(
        f"Elapsed time: {end_time - start_time} ns, fps={1e9 / (end_time - start_time)}"
    )
    pygame.image.save(surface, "maze.png")
    pygame.quit()


if __name__ == "__main__":
    maze = [
        "oooxxxxooo",
        "oxoxxxxoxo",
        "oxooooooxo",
        "oxxxxxxxxo",
        "oxooooooxo",
        "oxoxxxxoxo",
        "oooxxxxooo",
    ]

    create_maze_image_pygame(maze)
