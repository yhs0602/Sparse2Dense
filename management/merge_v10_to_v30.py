import wandb

from wandb_envs import WANDB_ENTITY, WANDB_PROJECT

if __name__ == "__main__":
    # Define the groups to move and the new group name
    groups_to_move = [
        ("v10-crossw2-sparse-2", "v30-crossw2-sparse-2"),
        ("v10-crossw2-dense-0", "v30-crossw2-dense-0"),
        ("v10-crossw2-dense-2", "v30-crossw2-dense-2"),
        ("v10-crossw2-transition-3000000-1", "v30-crossw2-transition-3000000-1"),
        ("v10-crossw2-transition-3000000-2", "v30-crossw2-transition-3000000-2"),
        ("v10-crossw2-transition-3000000-0", "v30-crossw2-transition-3000000-0"),
        ("v10-crossw2-sparse-0", "v30-crossw2-sparse-0"),
        ("v10-crossw2-dense-1", "v30-crossw2-dense-1"),
        ("v10-crossw2-sparse-1", "v30-crossw2-sparse-1"),
    ]

    # Move the runs to the new group
    api = wandb.Api()
    for group, new_group in groups_to_move:
        runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": group})
        for run in runs:
            run.group = new_group
            run.update()
            print(f"Moved run {run.name} from group {group} to {new_group}")

    print("All specified runs have been moved to the new group.")


# Moved run helpful-lake-406 from group v10-crossw2-sparse-2 to v30-crossw2-sparse-2
# Moved run electric-serenity-439 from group v10-crossw2-sparse-2 to v30-crossw2-sparse-2
# Moved run genial-water-446 from group v10-crossw2-sparse-2 to v30-crossw2-sparse-2
# Moved run young-vortex-465 from group v10-crossw2-sparse-2 to v30-crossw2-sparse-2
# Moved run dauntless-donkey-830 from group v10-crossw2-sparse-2 to v30-crossw2-sparse-2
# Moved run elated-snowflake-401 from group v10-crossw2-dense-0 to v30-crossw2-dense-0
# Moved run dulcet-voice-435 from group v10-crossw2-dense-0 to v30-crossw2-dense-0
# Moved run super-wildflower-442 from group v10-crossw2-dense-0 to v30-crossw2-dense-0
# Moved run avid-smoke-459 from group v10-crossw2-dense-0 to v30-crossw2-dense-0
# Moved run polished-vortex-468 from group v10-crossw2-dense-0 to v30-crossw2-dense-0
# Moved run earthy-flower-473 from group v10-crossw2-dense-0 to v30-crossw2-dense-0
# Moved run young-bee-475 from group v10-crossw2-dense-0 to v30-crossw2-dense-0
# Moved run misunderstood-bush-407 from group v10-crossw2-dense-2 to v30-crossw2-dense-2
# Moved run unique-eon-440 from group v10-crossw2-dense-2 to v30-crossw2-dense-2
# Moved run generous-wind-449 from group v10-crossw2-dense-2 to v30-crossw2-dense-2
# Moved run mild-violet-451 from group v10-crossw2-dense-2 to v30-crossw2-dense-2
# Moved run frosty-tree-466 from group v10-crossw2-dense-2 to v30-crossw2-dense-2
# Moved run earthy-oath-469 from group v10-crossw2-dense-2 to v30-crossw2-dense-2
# Moved run lunar-elevator-470 from group v10-crossw2-dense-2 to v30-crossw2-dense-2
# Moved run crisp-grass-405 from group v10-crossw2-transition-3000000-1 to v30-crossw2-transition-3000000-1
# Moved run bumbling-flower-438 from group v10-crossw2-transition-3000000-1 to v30-crossw2-transition-3000000-1
# Moved run faithful-galaxy-445 from group v10-crossw2-transition-3000000-1 to v30-crossw2-transition-3000000-1
# Moved run restful-plasma-464 from group v10-crossw2-transition-3000000-1 to v30-crossw2-transition-3000000-1
# Moved run dashing-flower-476 from group v10-crossw2-transition-3000000-1 to v30-crossw2-transition-3000000-1
# Moved run sleek-yogurt-408 from group v10-crossw2-transition-3000000-2 to v30-crossw2-transition-3000000-2
# Moved run celestial-energy-441 from group v10-crossw2-transition-3000000-2 to v30-crossw2-transition-3000000-2
# Moved run volcanic-butterfly-450 from group v10-crossw2-transition-3000000-2 to v30-crossw2-transition-3000000-2
# Moved run charmed-shape-467 from group v10-crossw2-transition-3000000-2 to v30-crossw2-transition-3000000-2
# Moved run comic-breeze-471 from group v10-crossw2-transition-3000000-2 to v30-crossw2-transition-3000000-2
# Moved run dark-sky-402 from group v10-crossw2-transition-3000000-0 to v30-crossw2-transition-3000000-0
# Moved run true-water-436 from group v10-crossw2-transition-3000000-0 to v30-crossw2-transition-3000000-0
# Moved run confused-waterfall-448 from group v10-crossw2-transition-3000000-0 to v30-crossw2-transition-3000000-0
# Moved run solar-cherry-460 from group v10-crossw2-transition-3000000-0 to v30-crossw2-transition-3000000-0
# Moved run fancy-water-474 from group v10-crossw2-transition-3000000-0 to v30-crossw2-transition-3000000-0
# Moved run dazzling-silence-400 from group v10-crossw2-sparse-0 to v30-crossw2-sparse-0
# Moved run glorious-sun-423 from group v10-crossw2-sparse-0 to v30-crossw2-sparse-0
# Moved run comfy-planet-447 from group v10-crossw2-sparse-0 to v30-crossw2-sparse-0
# Moved run twilight-snowball-458 from group v10-crossw2-sparse-0 to v30-crossw2-sparse-0
# Moved run amber-bush-472 from group v10-crossw2-sparse-0 to v30-crossw2-sparse-0
# Moved run faithful-serenity-404 from group v10-crossw2-dense-1 to v30-crossw2-dense-1
# Moved run magic-cosmos-437 from group v10-crossw2-dense-1 to v30-crossw2-dense-1
# Moved run fresh-dew-444 from group v10-crossw2-dense-1 to v30-crossw2-dense-1
# Moved run polished-gorge-463 from group v10-crossw2-dense-1 to v30-crossw2-dense-1
# Moved run glowing-galaxy-403 from group v10-crossw2-sparse-1 to v30-crossw2-sparse-1
# Moved run neat-cloud-434 from group v10-crossw2-sparse-1 to v30-crossw2-sparse-1
# Moved run avid-rain-443 from group v10-crossw2-sparse-1 to v30-crossw2-sparse-1
# Moved run trim-brook-462 from group v10-crossw2-sparse-1 to v30-crossw2-sparse-1
# All specified runs have been moved to the new group.
