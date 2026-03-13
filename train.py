"""
train.py — Mini World Model training (DreamerV3-lite)

Architecture:
  Encoder : obs → z  (CNN, 64-dim latent)
  RSSM    : (h_t, z_t, a_t) → h_{t+1}  (GRU, 256-dim)
  Decoder : (h_t, z_t) → obs_hat  (CNN transpose)

Loss = reconstruction (MSE/L1) + KL regularisation (optional, lightweight)

Designed to train in minutes on CPU for small datasets.
GPU accelerated automatically if available.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from world_gen import IMG_W, IMG_H, NUM_ACTIONS

# ─── Hyper-params ─────────────────────────────────────────────────────────────
LATENT_DIM  = 64
HIDDEN_DIM  = 256
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 3e-4
BETA_KL     = 0.1     # KL weight (set 0 to disable VAE)
CKPT        = "model.pt"
DATA_PATH   = "data/dataset.npz"


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TransitionDataset(Dataset):
    def __init__(self, path):
        d = np.load(path)
        # Normalize to [-1, 1]
        self.obs_t  = torch.from_numpy(d["obs_t"].astype(np.float32)  / 127.5 - 1.0).permute(0,3,1,2)
        self.acts   = torch.from_numpy(d["acts"].astype(np.int64))
        self.obs_t1 = torch.from_numpy(d["obs_t1"].astype(np.float32) / 127.5 - 1.0).permute(0,3,1,2)

    def __len__(self):
        return len(self.acts)

    def __getitem__(self, i):
        return self.obs_t[i], self.acts[i], self.obs_t1[i]


# ─── CNN Encoder ──────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """obs (3×H×W) → latent mean + log_var (LATENT_DIM each)"""
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 3×48×64 → 32×24×32
            nn.ELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # → 64×12×16
            nn.ELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # → 128×6×8
            nn.ELU(),
            nn.Flatten(),                                # → 128*6*8 = 6144
        )
        flat = 128 * (IMG_H // 8) * (IMG_W // 8)
        self.mu     = nn.Linear(flat, latent_dim)
        self.logvar = nn.Linear(flat, latent_dim)

    def forward(self, x):
        h       = self.net(x)
        mu      = self.mu(h)
        logvar  = self.logvar(h)
        return mu, logvar


def reparameterize(mu, logvar):
    if not mu.requires_grad:
        return mu
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# ─── Recurrent State Space Model (RSSM) ──────────────────────────────────────

class RSSM(nn.Module):
    """h_t = GRU(concat(z_t, a_t), h_{t-1})"""
    def __init__(self, latent_dim=LATENT_DIM, action_dim=NUM_ACTIONS, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, z, a_onehot, h):
        x = torch.cat([z, a_onehot], dim=-1)
        return self.gru(x, h)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)


# ─── CNN Decoder ──────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """(h, z) → obs_hat (3×H×W)"""
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        in_dim = latent_dim + hidden_dim
        flat   = 128 * (IMG_H // 8) * (IMG_W // 8)
        self.fc = nn.Linear(in_dim, flat)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # → 64×12×16
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # → 32×24×32
            nn.ELU(),
            nn.ConvTranspose2d(32, 3,  4, stride=2, padding=1),    # → 3×48×64
            nn.Tanh(),
        )
        self.h_shape = (128, IMG_H // 8, IMG_W // 8)

    def forward(self, h, z):
        x = torch.cat([h, z], dim=-1)
        x = self.fc(x)
        x = x.view(-1, *self.h_shape)
        return self.net(x)


# ─── Full World Model ─────────────────────────────────────────────────────────

class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(LATENT_DIM)
        self.rssm    = RSSM(LATENT_DIM, NUM_ACTIONS, HIDDEN_DIM)
        self.decoder = Decoder(LATENT_DIM, HIDDEN_DIM)

    def forward(self, obs_t, act, h=None):
        """
        obs_t: B×3×H×W  (current frame)
        act  : B         (action index)
        h    : B×HIDDEN  (recurrent state, None → zero-init)
        returns: obs_hat, h_new, mu, logvar
        """
        B = obs_t.size(0)
        device = obs_t.device

        if h is None:
            h = self.rssm.init_hidden(B, device)

        mu, logvar = self.encoder(obs_t)
        z          = reparameterize(mu, logvar)

        a_oh = F.one_hot(act, NUM_ACTIONS).float()
        h_new    = self.rssm(z, a_oh, h)

        obs_hat  = self.decoder(h_new, z)
        return obs_hat, h_new, mu, logvar

    def save(self, path=CKPT):
        torch.save({
            "model": self.state_dict(),
            "latent_dim": LATENT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_actions": NUM_ACTIONS,
        }, path)
        print(f"Saved → {path}")

    @classmethod
    def load(cls, path=CKPT, device="cpu"):
        ckpt  = torch.load(path, map_location=device)
        model = cls()
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model.to(device)


# ─── Loss ─────────────────────────────────────────────────────────────────────

def loss_fn(obs_hat, obs_t1, mu, logvar, beta=BETA_KL):
    recon = F.l1_loss(obs_hat, obs_t1)
    kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.item(), kl.item()


# ─── Training ─────────────────────────────────────────────────────────────────

def train(data_path=DATA_PATH, ckpt=CKPT, epochs=EPOCHS, batch_size=BATCH_SIZE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds    = TransitionDataset(data_path)
    n_val = max(1, int(len(ds) * 0.05))
    n_tr  = len(ds) - n_val
    tr_ds, val_ds = random_split(ds, [n_tr, n_val])

    tr_loader  = DataLoader(tr_ds,  batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model     = WorldModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=LR * 0.1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    best_val = float("inf")
    history  = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for obs_t, act, obs_t1 in tqdm(tr_loader, desc=f"Epoch {epoch:02d} train", leave=False):
            obs_t  = obs_t.to(device)
            act    = act.to(device)
            obs_t1 = obs_t1.to(device)

            obs_hat, _, mu, logvar = model(obs_t, act)
            loss, recon, kl        = loss_fn(obs_hat, obs_t1, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()

        scheduler.step()
        tr_loss /= len(tr_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_t, act, obs_t1 in val_loader:
                obs_t   = obs_t.to(device)
                act     = act.to(device)
                obs_t1  = obs_t1.to(device)
                obs_hat, _, mu, logvar = model(obs_t, act)
                loss, _, _ = loss_fn(obs_hat, obs_t1, mu, logvar)
                val_loss  += loss.item()
        val_loss /= len(val_loader)

        history.append((tr_loss, val_loss))
        print(f"Epoch {epoch:02d}/{epochs}  train={tr_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            model.save(ckpt)

    print(f"\nBest val loss: {best_val:.4f}")
    np.save("training_history.npy", np.array(history))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   type=str, default=DATA_PATH)
    parser.add_argument("--ckpt",   type=str, default=CKPT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train(data_path=args.data, ckpt=args.ckpt, epochs=args.epochs, batch_size=args.batch)
