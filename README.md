# Mini Dreamer
### Interactive World Model with Real-Time Online Learning

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)

Mini Dreamer is an experimental **interactive world model** that learns to predict the next observation **while you explore an environment in real time**.

The system continuously improves its predictions using **online training, replay buffers, and latent state dynamics**, while visualizing prediction quality and training metrics live.

This project is inspired by research on **world models and predictive representation learning**.

---

# Demo

The interface displays:

| Real Environment | Model Prediction |
|------------------|-----------------|
| Ground truth frame | Model's predicted next frame |

Additional panels show:

- prediction error heatmap
- learning metrics
- training curves
- latent state statistics

*(GIF demo recommended here)*

---

# Key Features

### Interactive exploration

Control an agent moving through a procedurally generated environment.

```
WASD movement
camera rotation
minimap visualization
```

---

### World model architecture

The model learns a predictive latent representation of the environment.

Pipeline:

```
observation_t
     ↓
encoder
     ↓
latent state (z)
     ↓
RSSM recurrent dynamics
     ↓
decoder
     ↓
predicted observation_{t+1}
```

The system predicts the next observation conditioned on the current state and action.

---

### Real-time online learning

The model updates continuously while the environment runs.

```
(obs_t, action, obs_t+1)
       ↓
   replay buffer
       ↓
   mini-batch sampling
       ↓
   gradient update
```

This allows observing **learning progress live**.

---

### Replay buffer training

Recent experience is stored in a circular replay buffer:

```
(obs_t, action, obs_t+1)
```

Mini-batches are sampled to perform gradient updates during gameplay.

---

### Prediction quality metrics

Multiple metrics measure prediction performance:

- **MSE** – mean squared error
- **MAE** – mean absolute error
- **PSNR** – peak signal-to-noise ratio
- **SSIM** – structural similarity
- **Histogram correlation**
- **RGB channel error**

Metrics are tracked over time to show model improvement.

---

### Live training visualization

The interface visualizes:

- prediction quality
- training loss
- gradient norm
- learning rate
- replay buffer size
- latent state statistics
- prediction error heatmap

This makes the learning dynamics directly observable.

---

### Pure Dream Mode

The model can run in **pure dream mode**, where it predicts future frames based only on its own predictions rather than real observations.

This reveals the internal generative dynamics of the world model.

---

# Controls

```
WASD   move
H      toggle error heatmap
L      toggle online learning
P      pure dream mode
R      reset latent state
F5     save checkpoint
Q      quit
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/mini-dreamer.git
cd mini-dreamer
```

Install dependencies:

```
pip install -r requirements.txt
```

Dependencies:

```
torch
numpy
pygame
```

---

# Running

Start the interactive simulation:

```
python play_learn.py
```

Optional parameters:

```
python play_learn.py --ckpt model.pt --seed 7 --scale 6
```

Arguments:

```
--ckpt    model checkpoint
--seed    environment seed
--scale   rendering scale
```

---

# Project Structure

```
mini-dreamer

play_learn.py
    interactive environment + online learning

train.py
    world model training utilities

world_gen.py
    procedural environment generation

checkpoints/
    saved model weights
```

---

# Training Objective

The model optimizes a combined objective:

```
L = reconstruction_loss + β * KL_divergence
```

Where

- reconstruction loss = L1 prediction error
- KL divergence regularizes the latent representation

---

# Purpose

This project serves as a **minimal experimental platform for exploring world models and predictive learning**.

It demonstrates:

- predictive representation learning
- recurrent latent state models
- replay buffer training
- real-time model improvement
- interactive ML visualization

---

# Future Work

Potential extensions:

- reinforcement learning agent
- curiosity-driven exploration
- transformer-based world models
- larger environments
- dataset logging and offline training

---

# License

MIT License

---

# Author

Tomasz Wietrzykowski
