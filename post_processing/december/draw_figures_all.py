import os
from matplotlib import pyplot as plt

from post_processing.december.draw_figures_entropy import (
    prepare_entropy_params,
    plot_entropy_groups,
    draw_entropy_figures,
    save_entropy_figure,
)
from post_processing.december.draw_normal_experiments import (
    prepare_normal_params,
    plot_groups_normal_experiments,
    plot_impl_normal_experiments,
)
from post_processing.december.draw_intrinsic_experiments import (
    prepare_intrinsic_params,
    plot_groups_intrinsics,
)

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_canonical_directory = os.path.realpath(current_directory)


def plot_groups(
    entropy_room_groups,
    entropy_groups_name,
    entropy_axis,
    entropy_window_size,
    normal_room_groups,
    normal_groups_name,
    room_axis,
    normal_window_size,
    intrinsic_room_groups,
    intrinsic_groups_name,
    intrinsic_window_size,
):
    axis_x_name = entropy_axis[0]
    axis_y_name = entropy_axis[1]

    plt.figure(figsize=(14, 8))

    if entropy_room_groups is not None:
        draw_entropy_figures(
            axis_x_name,
            axis_y_name,
            entropy_room_groups,
            entropy_groups_name,
            entropy_window_size,
        )
    else:
        print("No entropy data")
    if normal_room_groups is not None:
        plot_impl_normal_experiments(
            axis_x_name,
            axis_y_name,
            normal_room_groups,
            normal_groups_name,
            normal_window_size,
        )
    if intrinsic_room_groups is not None:
        plot_groups_intrinsics(
            axis_x_name,
            axis_y_name,
            intrinsic_room_groups,
            intrinsic_groups_name,
            intrinsic_window_size,
        )
    save_entropy_figure(
        axis_x_name,
        axis_y_name,
        entropy_groups_name,
        f"{current_canonical_directory}/figures/241214_all",
    )

    # axis_x_name = normal_axis[0]
    # axis_y_name = normal_axis[1]


def main():
    entropy_room_groups = None
    entropy_window_size = 80
    normal_room_groups = None
    normal_window_size = 80

    entropy_axises, entropy_room_groups, entropy_window_size = prepare_entropy_params(
        f"{current_canonical_directory}/../merged_entropy_241211"
    )
    intrinsic_axises, intrinsic_room_groups, intrinsic_window_size = (
        prepare_intrinsic_params(
            f"{current_canonical_directory}/../all_run_data241211-icm"
        )
    )
    normal_axises, normal_room_groups, normal_window_size = prepare_normal_params(
        f"{current_canonical_directory}/../merged_all_241211"
    )
    entropy_room_groups = None
    for room_axis in intrinsic_axises["room"]:
        plot_groups(
            entropy_room_groups,
            "room",
            room_axis,
            entropy_window_size,
            normal_room_groups,
            "room",
            room_axis,
            normal_window_size,
            intrinsic_room_groups,
            "room",
            intrinsic_window_size,
        )
        print(f"Plotted {room_axis}")


if __name__ == "__main__":
    main()
