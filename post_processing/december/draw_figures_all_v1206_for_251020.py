from matplotlib import pyplot as plt
from sympy.stats import entropy

from post_processing.december.draw_figures_entropy_v1206_for_251020 import (
    prepare_entropy_params,
    plot_entropy_groups,
    draw_entropy_figures,
    save_entropy_figure,
)
from post_processing.december.draw_normal_experiments_v1206_for_251020 import (
    prepare_normal_params,
    plot_groups_normal_experiments,
    plot_impl_normal_experiments,
)


def plot_groups(
    entropy_room_groups,
    entropy_groups_name,
    entropy_axis,
    entropy_window_size,
    normal_room_groups,
    normal_groups_name,
    room_axis,
    normal_window_size,
):
    axis_x_name = entropy_axis[0]
    axis_y_name = entropy_axis[1]

    plt.figure(figsize=(14, 8))

    draw_entropy_figures(
        axis_x_name,
        axis_y_name,
        entropy_room_groups,
        entropy_groups_name,
        entropy_window_size,
    )
    plot_impl_normal_experiments(
        axis_x_name,
        axis_y_name,
        normal_room_groups,
        normal_groups_name,
        normal_window_size,
    )
    save_entropy_figure(
        axis_x_name, axis_y_name, entropy_groups_name, "./figures/241206_all"
    )

    # axis_x_name = normal_axis[0]
    # axis_y_name = normal_axis[1]


def main():
    entropy_axises, entropy_room_groups, entropy_window_size = prepare_entropy_params()
    normal_axises, normal_room_groups, normal_window_size = prepare_normal_params(
        "../november/merged_241206"
    )
    for room_axis in entropy_axises["room"]:
        plot_groups(
            entropy_room_groups,
            "room",
            room_axis,
            entropy_window_size,
            normal_room_groups,
            "room",
            room_axis,
            normal_window_size,
        )
        print(f"Plotted {room_axis}")


if __name__ == "__main__":
    main()
