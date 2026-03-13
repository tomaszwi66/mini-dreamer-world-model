# Mini Dreamer
### Interactive World Model with Real-Time Online Learning

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)

Mini Dreamer is an experimental **interactive world model** that learns to predict the next observation **while you explore an environment in real time**.

The system continuously improves its predictions using **online training, replay buffers, and latent state dynamics**, while visualizing prediction quality and training metrics live.

This project demonstrates a minimal platform for exploring **predictive representation learning, world models, and online learning systems**.

---

# Demo

<video src="assets/demo.mp4" width="900" autoplay loop muted></video>

The interface displays:

| Real Environment | Model Prediction |
|------------------|-----------------|
| Ground truth frame | Model's predicted next frame |

Additional panels visualize:

- prediction error heatmap  
- learning metrics  
- training curves  
- replay buffer statistics  
- latent state information  

As the agent explores the environment, the model **gradually improves its predictions in real time**.

---

# Key Features

### Interactive exploration

Control an agent moving through a procedurally generated environment.

```
WASD movement
camera rotation
minimap visualization
```

The environment provides continuous observations used for training the world model.

---

### World model architecture

The system learns a predictive latent representation of the environment.

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

The model predicts the next observation conditioned on the current latent state and the executed action.

---

### Real-time online learning

The model updates continuously while the environment runs.

Training loop:

```
(obs_t, action, obs_t+1)
        ↓
    replay buffer
        ↓
    mini-batch sampling
        ↓
    gradient update
```

This allows observing **learning progress live during interaction**.

---

### Replay buffer

Recent experience is stored in a circular replay buffer:

```
(obs_t, action, obs_t+1)
```

Mini-batches sampled from the buffer are used for gradient updates during gameplay.

---

### Prediction quality metrics

The system evaluates prediction performance using several metrics:

- **MSE** - Mean Squared Error  
- **MAE** - Mean Absolute Error  
- **PSNR** - Peak Signal-to-Noise Ratio  
- **SSIM** - Structural Similarity  
- **Histogram Correlation**  
- **RGB Channel Error**

Metrics are tracked over time to measure **model improvement**.

---

### Live training visualization

The interface provides real-time insight into the learning process:

- prediction quality metrics
- error heatmaps
- training loss
- gradient norms
- learning rate
- replay buffer size
- latent state statistics

This makes the internal dynamics of the model visible during training.

---

### Pure Dream Mode

The system supports **Pure Dream Mode**, where the model generates predictions based solely on its own previous predictions rather than real observations.

This allows exploring the **internal generative dynamics of the learned world model**.

---

# Controls

```
WASD   move
H      toggle error heatmap
L      toggle online learning
P      pure dream mode
R      reset latent state
F5     save model checkpoint
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

# Running the Project

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
    interactive environment + online training

train.py
    world model training utilities

world_gen.py
    procedural environment generation

assets/
    demo video

checkpoints/
    saved model weights
```

---

# Training Objective

The model optimizes a combined objective:

```
L = reconstruction_loss + β · KL_divergence
```

Where:

- **reconstruction loss** = L1 prediction error  
- **KL divergence** regularizes the latent representation  

This objective encourages the model to learn a **compact predictive representation of the environment**.

---

# Purpose of the Project

Mini Dreamer is intended as a **minimal experimental platform for studying world models and predictive learning**.

The project demonstrates:

- predictive representation learning
- recurrent latent state models
- replay buffer training
- real-time online learning
- interactive ML visualization

The code is intentionally lightweight and designed for experimentation.

---

# Possible Extensions

Future directions may include:

- reinforcement learning agents
- curiosity-driven exploration
- transformer-based world models
- larger environments
- dataset logging for offline training
- multi-agent environments

---

# License

MIT License

---

# Author

Tomasz Wietrzykowski
