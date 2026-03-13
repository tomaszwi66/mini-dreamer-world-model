"""
eval.py — Quick visual evaluation: renders real vs dreamed comparison grid.
Saves eval_grid.png without needing pygame.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import math

from world_gen import generate_map, render, apply_action, MAP_SIZE, NUM_ACTIONS
from train import WorldModel, CKPT
from play import obs_to_tensor, tensor_to_img


def evaluate(ckpt=CKPT, seed=99, n_steps=16, save="eval_grid.png"):
    device = torch.device("cpu")
    model  = WorldModel.load(ckpt, device=device)

    grid, color_idx = generate_map(seed=seed)
    px, py  = MAP_SIZE / 2 + 0.5, MAP_SIZE / 2 + 0.5
    angle   = 0.5

    h = model.rssm.init_hidden(1, device)

    # Warm-up step
    real_img = render(grid, color_idx, px, py, angle)
    obs_t    = obs_to_tensor(real_img, device)
    with torch.no_grad():
        mu, _ = model.encoder(obs_t)
        a_oh  = F.one_hot(torch.tensor([0]), NUM_ACTIONS).float()
        h     = model.rssm(mu, a_oh, h)

    actions = [0, 3, 0, 3, 0, 0, 2, 0, 0, 3, 0, 0, 1, 2, 0, 0][:n_steps]

    reals   = []
    dreams  = []

    for act in actions:
        px, py, angle = apply_action(grid, px, py, angle, act)
        real_img      = render(grid, color_idx, px, py, angle)
        obs_t         = obs_to_tensor(real_img, device)

        with torch.no_grad():
            mu, _  = model.encoder(obs_t)
            act_t  = torch.tensor([act])
            a_oh   = F.one_hot(act_t, NUM_ACTIONS).float()
            h      = model.rssm(mu, a_oh, h)
            pred   = model.decoder(h, mu)
            dreamed = tensor_to_img(pred)

        reals.append(real_img)
        dreams.append(dreamed)

    # Plot grid
    cols = 8
    rows = (n_steps // cols) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 1.5))
    fig.patch.set_facecolor("#0d0d14")

    for i in range(n_steps):
        row_r = (i // cols) * 2
        row_d = row_r + 1
        col   = i % cols

        axes[row_r][col].imshow(reals[i])
        axes[row_r][col].axis("off")
        axes[row_r][col].set_title(f"Real {i}", fontsize=7, color="#aaa")

        axes[row_d][col].imshow(dreams[i])
        axes[row_d][col].axis("off")
        axes[row_d][col].set_title(f"Dream {i}", fontsize=7, color="#6ef")

    for ax in axes.flat:
        ax.set_facecolor("#0d0d14")

    plt.suptitle("Mini Dreamer — Real vs Dreamed", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=120, facecolor="#0d0d14")
    print(f"Saved → {save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   type=str, default=CKPT)
    parser.add_argument("--seed",   type=int, default=99)
    parser.add_argument("--steps",  type=int, default=16)
    parser.add_argument("--out",    type=str, default="eval_grid.png")
    args = parser.parse_args()
    evaluate(ckpt=args.ckpt, seed=args.seed, n_steps=args.steps, save=args.out)
