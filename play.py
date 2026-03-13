"""
play_learn.py — Play + learn in real-time.

The model fine-tunes itself on every step you take.
Watch metrics improve as you explore!

Controls: WASD=move, TAB=toggle, H=heatmap, R=reset_h
          P=pure_dream, L=toggle_learning, Q=quit
"""

import sys
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from world_gen import (
    generate_map, render, apply_action,
    MAP_SIZE, NUM_ACTIONS, IMG_W, IMG_H
)
from train import WorldModel, CKPT, LATENT_DIM, HIDDEN_DIM, BETA_KL


# ─── Replay Buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer storing recent transitions for online training."""

    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.obs_t  = deque(maxlen=capacity)
        self.acts   = deque(maxlen=capacity)
        self.obs_t1 = deque(maxlen=capacity)

    def add(self, obs_t: np.ndarray, action: int, obs_t1: np.ndarray):
        self.obs_t.append(obs_t.copy())
        self.acts.append(action)
        self.obs_t1.append(obs_t1.copy())

    def __len__(self):
        return len(self.acts)

    def sample(self, batch_size: int, device: torch.device):
        """Sample random mini-batch, return tensors."""
        n = len(self)
        batch_size = min(batch_size, n)
        indices = random.sample(range(n), batch_size)

        obs_t_batch  = np.stack([self.obs_t[i]  for i in indices])
        acts_batch   = np.array([self.acts[i]   for i in indices])
        obs_t1_batch = np.stack([self.obs_t1[i] for i in indices])

        obs_t_t  = torch.from_numpy(
            obs_t_batch.astype(np.float32) / 127.5 - 1.0
        ).permute(0, 3, 1, 2).to(device)

        acts_t = torch.from_numpy(acts_batch.astype(np.int64)).to(device)

        obs_t1_t = torch.from_numpy(
            obs_t1_batch.astype(np.float32) / 127.5 - 1.0
        ).permute(0, 3, 1, 2).to(device)

        return obs_t_t, acts_t, obs_t1_t


# ─── Online Trainer ──────────────────────────────────────────────────────────

class OnlineTrainer:
    """Handles online fine-tuning during gameplay."""

    def __init__(self, model: WorldModel, device: torch.device,
                 lr=1e-4, buffer_size=2000, batch_size=16,
                 train_every=1, min_buffer=32, beta_kl=0.05):
        self.model      = model
        self.device     = device
        self.batch_size = batch_size
        self.train_every = train_every
        self.min_buffer  = min_buffer
        self.beta_kl     = beta_kl

        self.buffer    = ReplayBuffer(capacity=buffer_size)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9999
        )

        # Stats
        self.train_steps   = 0
        self.total_loss_history   = deque(maxlen=500)
        self.recon_loss_history   = deque(maxlen=500)
        self.kl_loss_history      = deque(maxlen=500)
        self.grad_norm_history    = deque(maxlen=500)
        self.lr_history           = deque(maxlen=500)
        self.enabled       = True
        self.step_counter  = 0
        self.last_train_ms = 0.0

    def add_transition(self, obs_t, action, obs_t1):
        """Store new transition from gameplay."""
        self.buffer.add(obs_t, action, obs_t1)

    def maybe_train(self) -> dict:
        """Train one mini-batch if conditions met. Returns stats dict."""
        self.step_counter += 1

        if not self.enabled:
            return {}
        if len(self.buffer) < self.min_buffer:
            return {"status": f"buffering ({len(self.buffer)}/{self.min_buffer})"}
        if self.step_counter % self.train_every != 0:
            return {}

        return self._train_step()

    def _train_step(self) -> dict:
        """Execute one gradient step on sampled mini-batch."""
        self.model.train()

        obs_t, acts, obs_t1 = self.buffer.sample(self.batch_size, self.device)

        # Forward
        obs_hat, _, mu, logvar = self.model(obs_t, acts)

        # Loss
        recon = F.l1_loss(obs_hat, obs_t1)
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss  = recon + self.beta_kl * kl

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient norm (before clipping)
        grad_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.model.eval()
        self.train_steps += 1

        # Record stats
        stats = {
            "loss":      loss.item(),
            "recon":     recon.item(),
            "kl":        kl.item(),
            "grad_norm": grad_norm,
            "lr":        self.optimizer.param_groups[0]["lr"],
            "buffer":    len(self.buffer),
        }

        self.total_loss_history.append(stats["loss"])
        self.recon_loss_history.append(stats["recon"])
        self.kl_loss_history.append(stats["kl"])
        self.grad_norm_history.append(grad_norm)
        self.lr_history.append(stats["lr"])

        return stats


# ─── Metric computation ──────────────────────────────────────────────────────

def compute_metrics(real_img, dream_img):
    real  = real_img.astype(np.float32)
    dream = dream_img.astype(np.float32)

    mse = np.mean((real - dream) ** 2)
    mae = np.mean(np.abs(real - dream))
    psnr = 10.0 * math.log10(255.0**2 / max(mse, 1e-10))

    # SSIM
    g1 = np.mean(real, axis=2).flatten()
    g2 = np.mean(dream, axis=2).flatten()
    mu1, mu2 = g1.mean(), g2.mean()
    s1, s2   = g1.std(), g2.std()
    s12      = np.mean((g1 - mu1) * (g2 - mu2))
    C1, C2   = 6.5025, 58.5225
    ssim = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1**2+s2**2+C2))

    # Histogram correlation
    corrs = []
    for c in range(3):
        h1, _ = np.histogram(real[:,:,c], bins=32, range=(0,255))
        h2, _ = np.histogram(dream[:,:,c], bins=32, range=(0,255))
        h1 = h1.astype(np.float64); h1 /= h1.sum()+1e-8
        h2 = h2.astype(np.float64); h2 /= h2.sum()+1e-8
        cc = np.corrcoef(h1, h2)[0,1]
        corrs.append(cc if not np.isnan(cc) else 0.0)
    hist_corr = float(np.mean(corrs))

    # Per-channel
    r_err = np.mean(np.abs(real[:,:,0]-dream[:,:,0]))
    g_err = np.mean(np.abs(real[:,:,1]-dream[:,:,1]))
    b_err = np.mean(np.abs(real[:,:,2]-dream[:,:,2]))

    return {
        "mse": mse, "mae": mae, "psnr": psnr,
        "ssim": float(np.clip(ssim,-1,1)),
        "hist_corr": hist_corr,
        "r_err": r_err, "g_err": g_err, "b_err": b_err,
    }


class MetricTracker:
    def __init__(self, window=300):
        self.history = {k: deque(maxlen=window) for k in
                        ["mse","mae","psnr","ssim","hist_corr"]}
        self.cumulative_mae = 0.0
        self.total_steps = 0

    def update(self, m):
        for k in self.history:
            if k in m:
                self.history[k].append(m[k])
        self.cumulative_mae += m.get("mae", 0)
        self.total_steps += 1

    def avg(self, key, last_n=None):
        d = list(self.history.get(key, []))
        if not d: return 0.0
        if last_n: d = d[-last_n:]
        return sum(d)/len(d)

    def lifetime_mae(self):
        return self.cumulative_mae / max(self.total_steps, 1)

    def improvement(self, key, window=50):
        """Compare first vs last window entries. Positive = improved."""
        d = list(self.history.get(key, []))
        if len(d) < window * 2:
            return None
        first = sum(d[:window]) / window
        last  = sum(d[-window:]) / window
        if key in ("mae", "mse"):
            return first - last  # lower is better
        else:
            return last - first  # higher is better


# ─── Drawing ─────────────────────────────────────────────────────────────────

def obs_to_tensor(img, device):
    t = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
    return t.permute(2, 0, 1).unsqueeze(0).to(device)

def tensor_to_img(t):
    img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def draw_minimap(surface, grid, px, py, angle, ox, oy, size=120):
    cell = size // MAP_SIZE
    for r in range(MAP_SIZE):
        for c in range(MAP_SIZE):
            col = (60,60,60) if grid[r,c]==0 else (180,180,180)
            pygame.draw.rect(surface, col, (ox+c*cell, oy+r*cell, cell-1, cell-1))
    dx = int(ox + px*cell)
    dy = int(oy + py*cell)
    pygame.draw.circle(surface, (255,220,0), (dx,dy), max(2,cell//2))
    ex = int(dx + math.cos(angle)*cell*1.5)
    ey = int(dy + math.sin(angle)*cell*1.5)
    pygame.draw.line(surface, (255,80,80), (dx,dy), (ex,ey), 2)


def draw_graph(surface, data, x, y, w, h, color, label, font,
               min_val=None, max_val=None):
    pygame.draw.rect(surface, (30,30,40), (x,y,w,h))
    pygame.draw.rect(surface, (60,60,70), (x,y,w,h), 1)
    if len(data) < 2:
        lbl = font.render(f"{label}: …", True, (120,120,120))
        surface.blit(lbl, (x+4, y+2))
        return
    values = list(data)
    if min_val is None: min_val = min(values)
    if max_val is None: max_val = max(values)
    vr = max_val - min_val
    if vr < 1e-8: vr = 1.0
    pts = []
    for i, v in enumerate(values):
        px_ = x + int(i/max(len(values)-1,1)*(w-8)) + 4
        py_ = y + h - 4 - int((v-min_val)/vr*(h-16))
        py_ = max(y+2, min(y+h-2, py_))
        pts.append((px_, py_))
    if len(pts)>1:
        pygame.draw.lines(surface, color, False, pts, 2)
    lbl = font.render(f"{label}: {values[-1]:.2f}", True, color)
    surface.blit(lbl, (x+4, y+2))


def draw_bar(surface, value, x, y, w, h, label, font):
    value = max(0.0, min(1.0, value))
    pygame.draw.rect(surface, (30,30,40), (x,y,w,h))
    pygame.draw.rect(surface, (60,60,70), (x,y,w,h), 1)
    if value < 0.5:
        r, g = 255, int(value*2*255)
    else:
        r, g = int((1-value)*2*255), 255
    bw = int(value*(w-4))
    pygame.draw.rect(surface, (r,g,40), (x+2,y+2,bw,h-4))
    lbl = font.render(f"{label}: {value:.3f}", True, (220,220,220))
    surface.blit(lbl, (x+4, y+1))


def draw_heatmap(surface, real, dream, x, y, sc):
    diff = np.mean(np.abs(real.astype(np.float32)-dream.astype(np.float32)), axis=2)
    mx = max(diff.max(), 1.0)
    dn = (diff/mx*255).astype(np.uint8)
    hm = np.zeros((*dn.shape, 3), dtype=np.uint8)
    hm[:,:,0] = np.clip(dn*2, 0, 255)
    hm[:,:,1] = np.clip(dn, 0, 128)
    hs = pygame.surfarray.make_surface(np.transpose(hm, (1,0,2)))
    hs = pygame.transform.scale(hs, (IMG_W*sc, IMG_H*sc))
    surface.blit(hs, (x, y))


def improvement_arrow(val):
    """Return colored arrow string for improvement value."""
    if val is None:
        return "─", (120, 120, 120)
    if val > 0.5:
        return "▲▲", (50, 255, 50)
    elif val > 0.1:
        return "▲",  (100, 255, 100)
    elif val > -0.1:
        return "─",  (200, 200, 100)
    elif val > -0.5:
        return "▼",  (255, 150, 100)
    else:
        return "▼▼", (255, 80, 80)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_play(ckpt=CKPT, map_seed=7, scale=6):
    if not HAS_PYGAME:
        print("pip install pygame"); sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading {ckpt}…")
    model = WorldModel.load(ckpt, device=device)

    # ── Online trainer ──
    trainer = OnlineTrainer(
        model, device,
        lr=5e-5,            # lower than pretraining — fine-tune gently
        buffer_size=3000,
        batch_size=16,
        train_every=1,      # train on EVERY step
        min_buffer=32,
        beta_kl=0.03,
    )
    print(f"Online learning ENABLED (lr={trainer.optimizer.param_groups[0]['lr']})")

    grid, color_idx = generate_map(seed=map_seed)
    px, py = MAP_SIZE/2 + 0.5, MAP_SIZE/2 + 0.5
    angle  = 0.0

    VIEW_W = IMG_W * scale
    VIEW_H = IMG_H * scale
    PANEL_W = 340
    WIN_W = VIEW_W*2 + 20 + PANEL_W + 10
    WIN_H = max(VIEW_H + 200, 720)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Mini Dreamer — LIVE LEARNING")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)
    font_b = pygame.font.SysFont("monospace", 15, bold=True)

    h = model.rssm.init_hidden(1, device)
    real_img = render(grid, color_idx, px, py, angle)

    with torch.no_grad():
        obs_t = obs_to_tensor(real_img, device)
        mu, logvar = model.encoder(obs_t)
        z = mu
        a_oh = F.one_hot(torch.tensor([0], device=device), NUM_ACTIONS).float()
        h = model.rssm(z, a_oh, h)

    dreamed_img   = real_img.copy()
    tracker       = MetricTracker()
    current_metrics = {}
    train_stats   = {}
    last_mu_np    = mu.cpu().numpy().flatten()
    last_logvar_np = logvar.cpu().numpy().flatten()
    last_h_np     = h.cpu().numpy().flatten()

    action_map = {pygame.K_w:0, pygame.K_s:1, pygame.K_a:2, pygame.K_d:3}
    action_names = {0:"W fwd", 1:"S back", 2:"A left", 3:"D right"}
    last_action = "—"
    step_count = 0
    show_heatmap = True
    pure_dream = False
    dream_streak = 0
    inference_ms = 0
    running = True

    # For tracking before/after learning quality
    baseline_mae = None

    while running:
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_TAB:
                    pass  # reserved
                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap
                elif event.key == pygame.K_r:
                    h = model.rssm.init_hidden(1, device)
                    dream_streak = 0
                elif event.key == pygame.K_p:
                    pure_dream = not pure_dream
                    dream_streak = 0
                elif event.key == pygame.K_l:
                    trainer.enabled = not trainer.enabled
                    print(f"Learning: {'ON' if trainer.enabled else 'OFF'}")
                elif event.key == pygame.K_F5:
                    model.save(ckpt.replace(".pt", "_online.pt"))

        keys = pygame.key.get_pressed()
        for key, act in action_map.items():
            if keys[key]:
                action = act
                break

        if action is not None:
            last_action = action_names[action]
            prev_real = real_img.copy()

            px, py, angle = apply_action(grid, px, py, angle, action)
            real_img = render(grid, color_idx, px, py, angle)

            # ── Add to replay buffer ──
            trainer.add_transition(prev_real, action, real_img)

            # ── Neural prediction ──
            t0 = pygame.time.get_ticks()
            with torch.no_grad():
                act_t = torch.tensor([action], device=device)
                a_oh  = F.one_hot(act_t, NUM_ACTIONS).float()

                if pure_dream and dream_streak > 0:
                    obs_input = obs_to_tensor(dreamed_img, device)
                else:
                    obs_input = obs_to_tensor(real_img, device)

                mu, logvar = model.encoder(obs_input)
                z = mu
                h = model.rssm(z, a_oh, h)
                pred = model.decoder(h, z)
                dreamed_img = tensor_to_img(pred)

                last_mu_np = mu.cpu().numpy().flatten()
                last_logvar_np = logvar.cpu().numpy().flatten()
                last_h_np = h.cpu().numpy().flatten()
            inference_ms = pygame.time.get_ticks() - t0

            # ── Online training step ──
            train_stats = trainer.maybe_train()

            # ── Metrics ──
            current_metrics = compute_metrics(real_img, dreamed_img)
            tracker.update(current_metrics)

            if baseline_mae is None and tracker.total_steps >= 20:
                baseline_mae = tracker.avg("mae", 20)

            step_count += 1
            dream_streak += 1

        # ═══ DRAW ════════════════════════════════════════════════════════

        screen.fill((15, 15, 20))

        # Real + Dream views
        rs = pygame.surfarray.make_surface(np.transpose(real_img, (1,0,2)))
        rs = pygame.transform.scale(rs, (VIEW_W, VIEW_H))
        screen.blit(rs, (0, 0))

        ds = pygame.surfarray.make_surface(np.transpose(dreamed_img, (1,0,2)))
        ds = pygame.transform.scale(ds, (VIEW_W, VIEW_H))
        screen.blit(ds, (VIEW_W+20, 0))

        # Labels
        screen.blit(font_b.render("REAL", True, (200,200,200)), (4, VIEW_H+2))
        learn_status = "LEARNING" if trainer.enabled else "FROZEN"
        learn_color  = (50,255,50) if trainer.enabled else (255,100,100)
        screen.blit(font_b.render(
            f"DREAM [{learn_status}]", True, learn_color
        ), (VIEW_W+24, VIEW_H+2))

        # Heatmap
        if show_heatmap and step_count > 0:
            hm_y = VIEW_H + 22
            screen.blit(font.render("ERROR HEATMAP", True, (200,160,60)),
                        (VIEW_W+24, hm_y))
            draw_heatmap(screen, real_img, dreamed_img,
                         VIEW_W+24, hm_y+16, scale//2)

        # Minimap
        draw_minimap(screen, grid, px, py, angle,
                     VIEW_W//2 - 60, VIEW_H+22)

        # ═══ METRICS PANEL ═══════════════════════════════════════════════

        PX = VIEW_W*2 + 30
        cy = 8
        bw = PANEL_W - 10

        # ── Prediction Quality ──
        screen.blit(font_b.render("══ PREDICTION ══", True, (255,200,80)),
                    (PX, cy)); cy += 22

        if current_metrics:
            mlines = [
                (f"MSE:       {current_metrics['mse']:.1f}",    (255,120,120)),
                (f"MAE:       {current_metrics['mae']:.1f}",    (255,150,100)),
                (f"PSNR:      {current_metrics['psnr']:.1f} dB",(100,255,150)),
                (f"SSIM:      {current_metrics['ssim']:.4f}",   (100,200,255)),
                (f"Hist Corr: {current_metrics['hist_corr']:.4f}",(200,180,255)),
                (f"RGB err:   {current_metrics['r_err']:.0f}/{current_metrics['g_err']:.0f}/{current_metrics['b_err']:.0f}",
                 (200,200,200)),
            ]
        else:
            mlines = [("(move to see metrics)", (120,120,120))]

        for text, col in mlines:
            screen.blit(font.render(text, True, col), (PX, cy)); cy += 15
        cy += 6

        # Quality bars
        for key, label in [("ssim","SSIM"),("hist_corr","Hist")]:
            draw_bar(screen, current_metrics.get(key, 0),
                     PX, cy, bw, 16, label, font)
            cy += 20
        cy += 4

        # ── Improvement tracking ──
        screen.blit(font_b.render("══ IMPROVEMENT ══", True, (100,255,200)),
                    (PX, cy)); cy += 20

        for key, label, better_dir in [
            ("mae","MAE","lower"), ("psnr","PSNR","higher"), ("ssim","SSIM","higher")
        ]:
            imp = tracker.improvement(key)
            arrow, acol = improvement_arrow(
                imp if better_dir == "higher" else (imp if imp is None else imp)
            )
            current_avg = tracker.avg(key, 20)
            txt = f"{arrow} {label}: {current_avg:.2f}"
            if imp is not None:
                txt += f" (Δ={imp:+.2f})"
            screen.blit(font.render(txt, True, acol), (PX, cy)); cy += 15

        if baseline_mae is not None:
            current_mae = tracker.avg("mae", 20)
            pct = (baseline_mae - current_mae) / max(baseline_mae, 1) * 100
            col = (50,255,50) if pct > 0 else (255,80,80)
            screen.blit(font.render(
                f"vs baseline: {pct:+.1f}%", True, col
            ), (PX, cy))
        cy += 20

        # ── Sparklines ──
        gh, gw = 36, bw

        draw_graph(screen, tracker.history["psnr"], PX, cy, gw, gh,
                   (100,255,150), "PSNR", font, 10, 40)
        cy += gh + 3

        draw_graph(screen, tracker.history["mae"], PX, cy, gw, gh,
                   (255,150,100), "MAE", font, 0, 60)
        cy += gh + 3

        draw_graph(screen, tracker.history["ssim"], PX, cy, gw, gh,
                   (100,200,255), "SSIM", font, 0.5, 1.0)
        cy += gh + 8

        # ══ TRAINING PANEL ═══════════════════════════════════════════════

        screen.blit(font_b.render("══ ONLINE TRAIN ══", True, (255,180,50)),
                    (PX, cy)); cy += 20

        if trainer.train_steps > 0:
            tlines = [
                f"Train steps:  {trainer.train_steps}",
                f"Buffer:       {len(trainer.buffer)}/{trainer.buffer.capacity}",
                f"LR:           {trainer.optimizer.param_groups[0]['lr']:.2e}",
            ]
            if train_stats.get("loss"):
                tlines += [
                    f"Loss:         {train_stats['loss']:.4f}",
                    f"  recon:      {train_stats['recon']:.4f}",
                    f"  kl:         {train_stats['kl']:.4f}",
                    f"Grad norm:    {train_stats.get('grad_norm',0):.2f}",
                ]
        elif "status" in train_stats:
            tlines = [train_stats["status"]]
        else:
            tlines = ["waiting for data…"]

        for t in tlines:
            screen.blit(font.render(t, True, (180,180,140)), (PX, cy)); cy += 15
        cy += 4

        # Training loss graph
        if len(trainer.recon_loss_history) > 2:
            draw_graph(screen, trainer.recon_loss_history, PX, cy, gw, gh,
                       (255,200,80), "Train L1", font, 0, None)
            cy += gh + 3
            draw_graph(screen, trainer.grad_norm_history, PX, cy, gw, gh,
                       (200,100,255), "GradNorm", font, 0, None)
            cy += gh + 6

        # ── Latent info ──
        screen.blit(font_b.render("══ LATENT ══", True, (180,180,180)),
                    (PX, cy)); cy += 18
        mu_abs = np.mean(np.abs(last_mu_np))
        var_m  = np.mean(np.exp(last_logvar_np))
        h_norm = np.linalg.norm(last_h_np)
        h_act  = int(np.sum(np.abs(last_h_np)>0.1))

        for t in [
            f"z: |μ|={mu_abs:.2f}  σ²={var_m:.3f}",
            f"h: ‖h‖={h_norm:.1f}  active={h_act}/{len(last_h_np)}",
        ]:
            screen.blit(font.render(t, True, (180,170,220)), (PX, cy)); cy += 15
        cy += 8

        # ── System ──
        for t in [
            f"Step:{step_count} Act:{last_action} Inf:{inference_ms}ms",
            f"FPS:{clock.get_fps():.0f} Dream:{'PURE' if pure_dream else 'ground'}",
            f"Pos:({px:.1f},{py:.1f}) Ang:{math.degrees(angle):.0f}°",
        ]:
            screen.blit(font.render(t, True, (120,120,120)), (PX, cy)); cy += 15

        # Controls
        for i, line in enumerate([
            "WASD=move H=heat L=learn R=reset",
            "P=pure_dream F5=save Q=quit"
        ]):
            screen.blit(font.render(line, True, (80,80,100)),
                        (4, WIN_H-36+i*16))

        pygame.display.flip()
        clock.tick(30)

    # ── Session summary ──
    print("\n" + "="*55)
    print("  SESSION SUMMARY")
    print("="*55)
    print(f"  Steps played:      {step_count}")
    print(f"  Online train steps:{trainer.train_steps}")
    print(f"  Buffer size:       {len(trainer.buffer)}")
    print(f"  Lifetime MAE:      {tracker.lifetime_mae():.1f}")
    print(f"  Final MAE (20):    {tracker.avg('mae', 20):.1f}")
    print(f"  Final PSNR (20):   {tracker.avg('psnr', 20):.1f} dB")
    print(f"  Final SSIM (20):   {tracker.avg('ssim', 20):.4f}")
    if baseline_mae is not None:
        final_mae = tracker.avg("mae", 20)
        pct = (baseline_mae - final_mae) / max(baseline_mae, 1) * 100
        print(f"  MAE improvement:   {pct:+.1f}% vs start")
    print("="*55)

    # Auto-save
    save_path = ckpt.replace(".pt", "_online.pt")
    model.save(save_path)
    print(f"  Model saved → {save_path}")

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  type=str, default=CKPT)
    parser.add_argument("--seed",  type=int, default=7)
    parser.add_argument("--scale", type=int, default=6)
    args = parser.parse_args()

    run_play(ckpt=args.ckpt, map_seed=args.seed, scale=args.scale)