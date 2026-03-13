"""
world_gen.py — Procedural raycasting world generator (Wolfenstein-style)
Generates small maps with guaranteed connectivity and renders first-person views.
"""

import numpy as np
import math
import random
from typing import Tuple, List

# ─── Raycasting config ────────────────────────────────────────────────────────
IMG_W   = 64   # render width  (keep small for fast training)
IMG_H   = 48   # render height
FOV     = math.pi / 3   # 60°
HALF_FOV = FOV / 2
MAX_DEPTH = 12.0
NUM_RAYS = IMG_W

# ─── Map config ───────────────────────────────────────────────────────────────
MAP_SIZE = 12   # square map (inner 10×10, outer wall)
WALL_DENSITY = 0.22

# ─── Colors ───────────────────────────────────────────────────────────────────
FLOOR_COLOR   = np.array([40,  40,  40],  dtype=np.uint8)
CEILING_COLOR = np.array([80,  100, 120], dtype=np.uint8)
WALL_COLORS   = [
    np.array([180, 60,  60],  dtype=np.uint8),   # red brick
    np.array([60,  140, 80],  dtype=np.uint8),   # mossy stone
    np.array([160, 130, 60],  dtype=np.uint8),   # sandstone
    np.array([80,  80,  160], dtype=np.uint8),   # blue dungeon
    np.array([120, 80,  40],  dtype=np.uint8),   # wood
]

def generate_map(seed: int = None) -> Tuple[np.ndarray, int]:
    """Generate a random map with outer walls and guaranteed open area.
    Returns (grid, wall_color_idx).
    grid: MAP_SIZE×MAP_SIZE, 1=wall, 0=floor
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    grid = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    # Outer walls
    grid[0, :] = 1; grid[-1, :] = 1
    grid[:, 0] = 1; grid[:, -1] = 1

    # Random inner walls — leave center 3×3 always open for spawn
    cx, cy = MAP_SIZE // 2, MAP_SIZE // 2
    for r in range(1, MAP_SIZE - 1):
        for c in range(1, MAP_SIZE - 1):
            if abs(r - cx) <= 1 and abs(c - cy) <= 1:
                continue
            if random.random() < WALL_DENSITY:
                grid[r, c] = 1

    # Ensure connectivity with simple flood fill check; retry bad maps
    if not _is_connected(grid):
        return generate_map(seed=(seed or 0) + 1000)

    color_idx = random.randint(0, len(WALL_COLORS) - 1)
    return grid, color_idx


def _is_connected(grid: np.ndarray) -> bool:
    """BFS from center; all floor cells must be reachable."""
    rows, cols = grid.shape
    start = (rows // 2, cols // 2)
    if grid[start] == 1:
        return False
    visited = set()
    queue = [start]
    while queue:
        r, c = queue.pop()
        if (r, c) in visited: continue
        visited.add((r, c))
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr,nc] == 0:
                queue.append((nr, nc))
    total_floor = int((grid == 0).sum())
    return len(visited) == total_floor


def get_floor_cells(grid: np.ndarray) -> List[Tuple[int,int]]:
    """Return list of (row, col) floor cells."""
    rows, cols = grid.shape
    return [(r, c) for r in range(rows) for c in range(cols) if grid[r, c] == 0]


# ─── Raycaster ────────────────────────────────────────────────────────────────

def cast_ray(grid, px, py, angle):
    """DDA ray cast. Returns distance to wall and hit-side (0=NS, 1=EW)."""
    ray_cos = math.cos(angle)
    ray_sin = math.sin(angle)

    # Avoid division by zero
    eps = 1e-9
    ray_cos = ray_cos if abs(ray_cos) > eps else eps
    ray_sin = ray_sin if abs(ray_sin) > eps else eps

    # DDA setup
    map_x, map_y = int(px), int(py)
    delta_x = abs(1.0 / ray_cos)
    delta_y = abs(1.0 / ray_sin)

    step_x = 1 if ray_cos > 0 else -1
    step_y = 1 if ray_sin > 0 else -1

    side_x = (map_x + 1 - px) * delta_x if ray_cos > 0 else (px - map_x) * delta_x
    side_y = (map_y + 1 - py) * delta_y if ray_sin > 0 else (py - map_y) * delta_y

    side = 0
    for _ in range(64):
        if side_x < side_y:
            side_x += delta_x
            map_x  += step_x
            side = 0
        else:
            side_y += delta_y
            map_y  += step_y
            side = 1
        if 0 <= map_x < MAP_SIZE and 0 <= map_y < MAP_SIZE:
            if grid[map_y, map_x] == 1:
                break
        else:
            break

    if side == 0:
        dist = (map_x - px + (1 - step_x) / 2) / ray_cos
    else:
        dist = (map_y - py + (1 - step_y) / 2) / ray_sin

    return max(0.01, abs(dist)), side


def render(grid, color_idx, px, py, angle) -> np.ndarray:
    """Render first-person view. Returns H×W×3 uint8 RGB image."""
    img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    wall_color = WALL_COLORS[color_idx]
    half_h = IMG_H // 2

    # Floor / ceiling
    img[:half_h, :] = CEILING_COLOR
    img[half_h:, :] = FLOOR_COLOR

    for col in range(NUM_RAYS):
        ray_angle = angle - HALF_FOV + (col / NUM_RAYS) * FOV
        dist, side = cast_ray(grid, px, py, ray_angle)

        # Fish-eye correction
        dist_corrected = dist * math.cos(ray_angle - angle)
        dist_corrected = max(0.1, dist_corrected)

        wall_height = int(IMG_H / dist_corrected)
        wall_top    = max(0, half_h - wall_height // 2)
        wall_bottom = min(IMG_H, half_h + wall_height // 2)

        # Shade by distance and side
        shade = max(0.2, 1.0 - dist_corrected / MAX_DEPTH)
        if side == 1:
            shade *= 0.7

        color = (wall_color * shade).clip(0, 255).astype(np.uint8)
        img[wall_top:wall_bottom, col] = color

    return img


# ─── Action mechanics ─────────────────────────────────────────────────────────

MOVE_SPEED = 0.15
TURN_SPEED = math.pi / 12   # 15° per step

ACTIONS = {
    0: "W",   # forward
    1: "S",   # backward
    2: "A",   # turn left
    3: "D",   # turn right
}
NUM_ACTIONS = len(ACTIONS)


def apply_action(grid, px, py, angle, action: int):
    """Apply action, return (new_px, new_py, new_angle). Collision safe."""
    dx, dy, da = 0.0, 0.0, 0.0
    if action == 0:   # forward
        dx =  math.cos(angle) * MOVE_SPEED
        dy =  math.sin(angle) * MOVE_SPEED
    elif action == 1: # backward
        dx = -math.cos(angle) * MOVE_SPEED
        dy = -math.sin(angle) * MOVE_SPEED
    elif action == 2: # turn left
        da = -TURN_SPEED
    elif action == 3: # turn right
        da =  TURN_SPEED

    new_angle = angle + da
    new_px    = px + dx
    new_py    = py + dy

    # Collision detection (with small margin)
    margin = 0.25
    if grid[int(new_py), int(px + dx)] == 0:
        px = new_px
    if grid[int(py + dy), int(new_px)] == 0:
        py = new_py

    return px, py, new_angle % (2 * math.pi)


# ─── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    grid, cidx = generate_map(seed=42)
    print("Map:")
    for row in grid:
        print("".join("█" if c else "." for c in row))

    px, py = MAP_SIZE / 2, MAP_SIZE / 2
    angle = 0.0
    img = render(grid, cidx, px, py, angle)

    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.title(f"Raycaster test — color {cidx}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("test_render.png", dpi=100)
    print("Saved test_render.png")
