import os
from matplotlib import pyplot as plt
import matplotlib as mpl


from post_processing.december.draw_figures_entropy import (
    prepare_entropy_params,
    draw_entropy_figures,
    save_entropy_figure,
)
from post_processing.december.draw_normal_experiments import (
    prepare_normal_params,
    plot_impl_normal_experiments,
)
from post_processing.december.draw_intrinsic_experiments import (
    prepare_intrinsic_params,
    plot_groups_intrinsics,
)

current_file_path = __file__
current_directory = os.path.dirname(current_file_path)
current_canonical_directory = os.path.realpath(current_directory)

mpl.rcParams.update(
    {
        # "figure.figsize": (18, 5),  # subplot enlarge
        # "axes.titlesize": 12,
        # "axes.labelsize": 11,
        # "legend.fontsize": 10,
        # "pdf.fonttype": 42,  #  Preserve font type
        # "ps.fonttype": 42,
    }
)

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
COLORS = plt.get_cmap("tab10").colors  #  Contrast appropriate


def move_legend_outside(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0.0,
            frameon=False,
        )


def restyle_axes(ax):
    # Reduce excessive red dominance: uniform thickness, uniform zorder
    for i, line in enumerate(ax.get_lines()):
        # line.set_linewidth(1.8)
        line.set_zorder(2)
        # line.set_color(COLORS[i % len(COLORS)])
        # line.set_linestyle(LINESTYLES[i % len(LINESTYLES)])
    # Reduce CI(alpha) to improve readability: apply to fill_between artists
    for coll in ax.collections:
        try:
            coll.set_alpha(0.15)  # Reduce CI(alpha) to improve readability
        except Exception:
            pass


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
        try:
            plot_impl_normal_experiments(
                axis_x_name,
                axis_y_name,
                normal_room_groups,
                normal_groups_name,
                normal_window_size,
            )
        except KeyError as e:
            print(f"KeyError | AssertionError: {axis_x_name}, {axis_y_name}, {e}")
    if intrinsic_room_groups is not None:
        plot_groups_intrinsics(
            axis_x_name,
            axis_y_name,
            intrinsic_room_groups,
            intrinsic_groups_name,
            intrinsic_window_size,
        )

    # axes = fig.get_axes()
    # for ax in axes:
    #     restyle_axes(ax)
    #     move_legend_outside(ax)
    # fig.tight_layout()

    save_entropy_figure(
        axis_x_name,
        axis_y_name,
        entropy_groups_name,
        f"{current_canonical_directory}/figures/241214_all-icm_ir-251019",
        do_title=False,
        do_legend=True,
    )

    # fig.savefig(
    #     f"{current_canonical_directory}/figures/241214_all-icm_ir-251019.pdf",
    #     format="pdf",
    #     bbox_inches="tight",
    # )

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
            f"{current_canonical_directory}/../all_run_data241214-icm-with-reward"
        )
    )
    normal_axises, normal_room_groups, normal_window_size = prepare_normal_params(
        f"{current_canonical_directory}/../merged_all_241214-s2d-with-reward"
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
