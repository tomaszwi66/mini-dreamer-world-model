"""
collect_data.py — Systematic data collection for world model training.

Strategy: boustrophedon (snake) walk covering entire map + random exploration.
Each step: render current frame → take action → render next frame.
Saves (obs_t, action, obs_t1) triples as compressed numpy arrays.
"""

import numpy as np
import math
import random
import os
import argparse
from tqdm import tqdm

from world_gen import (
    generate_map, render, apply_action, get_floor_cells,
    MAP_SIZE, NUM_ACTIONS, MOVE_SPEED, IMG_W, IMG_H
)

# ─── Config ───────────────────────────────────────────────────────────────────
NUM_MAPS          = 20       # how many different maps to collect from
STEPS_PER_MAP     = 2000     # transitions per map
COVERAGE_FRACTION = 0.5      # fraction of steps that are systematic
OUT_DIR           = "data"


def systematic_walk(grid, px, py, angle, n_steps):
    """Boustrophedon-ish walk: try to visit all floor cells systematically."""
    floor_cells = get_floor_cells(grid)
    random.shuffle(floor_cells)

    trajectory = []
    cell_queue = list(floor_cells)

    for _ in range(n_steps):
        obs = render(grid, 0, px, py, angle)  # color_idx doesn't matter here — fixed per map

        # Pick a target cell
        if cell_queue:
            tr, tc = cell_queue[0]
            tx, ty = tc + 0.5, tr + 0.5

            dx, dy = tx - px, ty - py
            dist   = math.sqrt(dx * dx + dy * dy)

            if dist < 0.4:
                cell_queue.pop(0)
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                target_angle = math.atan2(dy, dx) % (2 * math.pi)
                angle_diff = (target_angle - angle + math.pi) % (2 * math.pi) - math.pi

                if abs(angle_diff) > 0.25:
                    action = 3 if angle_diff > 0 else 2   # turn right/left
                else:
                    action = 0  # move forward
        else:
            action = random.randint(0, NUM_ACTIONS - 1)

        trajectory.append((obs, action, px, py, angle))
        px, py, angle = apply_action(grid, px, py, angle, action)

    return trajectory, px, py, angle


def random_walk(grid, px, py, angle, n_steps):
    """Pure random walk for exploration diversity."""
    trajectory = []
    for _ in range(n_steps):
        obs    = render(grid, 0, px, py, angle)
        action = random.randint(0, NUM_ACTIONS - 1)
        trajectory.append((obs, action, px, py, angle))
        px, py, angle = apply_action(grid, px, py, angle, action)
    return trajectory, px, py, angle


def collect(num_maps=NUM_MAPS, steps_per_map=STEPS_PER_MAP, out_dir=OUT_DIR, seed=0):
    os.makedirs(out_dir, exist_ok=True)

    all_obs_t  = []
    all_acts   = []
    all_obs_t1 = []

    total = num_maps * steps_per_map
    pbar  = tqdm(total=total, desc="Collecting")

    for map_id in range(num_maps):
        grid, color_idx = generate_map(seed=seed + map_id)

        # Spawn at center
        px = MAP_SIZE / 2 + 0.5
        py = MAP_SIZE / 2 + 0.5
        angle = random.uniform(0, 2 * math.pi)

        sys_steps = int(steps_per_map * COVERAGE_FRACTION)
        rnd_steps = steps_per_map - sys_steps

        traj_sys, px, py, angle = systematic_walk(grid, px, py, angle, sys_steps)
        traj_rnd, px, py, angle = random_walk(grid, px, py, angle, rnd_steps)
        traj = traj_sys + traj_rnd

        for i in range(len(traj) - 1):
            obs_t, act, cpx, cpy, cang = traj[i]
            # Render t+1 with updated position (next entry)
            _, _, npx, npy, nang = traj[i + 1]
            obs_t1 = render(grid, color_idx, npx, npy, nang)

            all_obs_t.append(obs_t.astype(np.uint8))
            all_acts.append(act)
            all_obs_t1.append(obs_t1.astype(np.uint8))

        pbar.update(steps_per_map)

    pbar.close()

    obs_t  = np.stack(all_obs_t,  axis=0)   # N × H × W × 3
    acts   = np.array(all_acts,   dtype=np.uint8)
    obs_t1 = np.stack(all_obs_t1, axis=0)

    print(f"\nDataset: {len(acts):,} transitions")
    print(f"obs shape: {obs_t.shape}  dtype: {obs_t.dtype}")

    np.savez_compressed(
        os.path.join(out_dir, "dataset.npz"),
        obs_t=obs_t, acts=acts, obs_t1=obs_t1
    )
    print(f"Saved → {out_dir}/dataset.npz")
    return obs_t, acts, obs_t1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps",  type=int, default=NUM_MAPS,      help="Number of maps")
    parser.add_argument("--steps", type=int, default=STEPS_PER_MAP, help="Steps per map")
    parser.add_argument("--seed",  type=int, default=0)
    parser.add_argument("--out",   type=str, default=OUT_DIR)
    args = parser.parse_args()

    collect(num_maps=args.maps, steps_per_map=args.steps, out_dir=args.out, seed=args.seed)
