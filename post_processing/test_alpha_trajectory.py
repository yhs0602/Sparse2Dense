from post_processing.alpha_trajectory import create_trajectory_image


def main():
    create_trajectory_image(
        [
            (0, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 0, 0),
            (5, 0, 0),
            (6, 0, 0),
            (7, 0, 0),
            (8, 0, 0),
            (9, 0, 0),
        ],
        "test.png",
        goals=[(0.3, 0.3)],
    )


if __name__ == "__main__":
    main()
