# Reinforcement Learning: Q-Learning & Deep Q-Networks

This repository contains two reinforcement learning projects developed as part of a "Principles of Autonomy and Decision Making" course assignment.

The project explores two approaches to learning action-value functions:

- **Part 1 — Q-Learning:** Tabular Q-learning in a custom discrete grid-world.
- **Part 2 — Deep Q-Network (DQN):** Neural-network-based Q-learning in a separate continuous-state maze environment.

Both projects include training, policy evaluation, visualization, and learned policy animations.

---

## Repository Structure

```text
.
├── Part1_Q_Learning/
│   ├── main.py
│   ├── env.py
│   ├── Q_learning.py
│   ├── utils.py
│   ├── imgs/ - ignored by git
│   └── runs/ - ignored by git
│
├── Part2_DQN/
│   ├── main.py
│   ├── env.py
│   ├── DQN_model.py
│   ├── utils.py
│   └── runs/ - ignored by git
│
└── README.md
```

---

# Part 1 — Q-Learning

## Overview

The first part implements a custom **discrete grid-world environment** and trains an agent using tabular Q-learning.

The agent navigates through a grid towards a goal while avoiding dangerous states and puddles.

Two variants of the environment are implemented:

1. **Basic / deterministic environment**
2. **Stochastic environment**

## Environment

The agent's state is represented by its grid position:

```text
state = (row, column)
```

Four discrete actions are available:

| Action | Direction |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Right |
| 3 | Left |

The environment contains:

- **Goal** — reaching the goal terminates the episode with a positive reward.
- **Danger states** — entering a danger state terminates the episode with a negative reward.
- **Puddles** — entering a puddle produces a negative reward and affects the stochastic dynamics.
- **Empty cells** — provide a small negative step reward.
- **Grid boundaries** — attempting to leave the grid produces a penalty and keeps the agent in its previous position.

---

## Deterministic Environment

In the basic environment, the selected action is always executed:

```text
P(selected action) = 1.0
```

If the agent selects `Right`, the environment executes `Right` unless movement is blocked by a boundary.

---

## Stochastic Environment

The stochastic variant introduces uncertainty into action execution.

Initially, the probability of executing the selected action is:

```text
P = 0.9
```

The remaining probability is distributed equally among the other three actions.

For a selected action `a`:

```text
P(real action = a) = P
P(real action != a) = (1 - P) / 3
```

### Puddle Effect

When the agent steps into a puddle, the probability of executing the selected action decreases:

```text
P = P - 0.1
```

The minimum probability is:

```text
P = 0.5
```

Therefore:

```text
0.9 → 0.8 → 0.7 → 0.6 → 0.5
```

This creates increasingly stochastic movement after the agent encounters puddles.

---

## Q-Learning

The agent learns an action-value function:

```text
Q(s, a)
```

The Q-values are updated according to the Q-learning update rule:

```text
Q(s,a) ← Q(s,a) +
         α [r + γ max Q(s',a') − Q(s,a)]
```

where:

- `α` — learning rate
- `γ` — discount factor
- `r` — received reward
- `s` — current state
- `a` — selected action
- `s'` — next state

## Exploration

An **epsilon-greedy exploration strategy** is used.

With probability `ε`, the agent selects a random action. Otherwise, it selects the action with the highest estimated Q-value.

The exploration rate decays during training:

```text
ε = max(ε_min, ε × ε_decay)
```

## Results

The project generates visualizations of the learned policies, Q-values, and training rewards.

---

# Part 2 — Deep Q-Network

## Overview

The second part uses a **separate continuous-state maze environment** and implements a Deep Q-Network.

Unlike Part 1, where the agent's position is represented by a discrete grid state, the agent in this environment has a continuous 2D position:

```text
state = (x, y)

x ∈ [0, 1]
y ∈ [0, 1]
```

The environment contains a goal region, danger zones, and maze walls.

## Continuous Maze Environment

The agent has four discrete actions:

| Action | Direction |
|---|---|
| 0 | Up |
| 1 | Down |
| 2 | Left |
| 3 | Right |

The agent moves through the environment using a fixed step size.

The environment includes:

- A circular goal region
- Multiple danger zones
- Maze walls
- A continuous 2D state space

## Reward Structure

The environment uses:

| Event | Reward |
|---|---:|
| Reach goal | `+20` |
| Enter danger zone | `-50` |
| Collide with wall | `-5` |
| Normal movement | Negative distance-based reward |

The normal movement reward is based on the Manhattan distance to the goal:

```text
d = |x - x_goal| + |y - y_goal|
```

Additional reward shaping is applied near danger zones to encourage the agent to maintain a safe distance.

## Deep Q-Network

Instead of storing Q-values in a table, a neural network approximates the Q-function.

The network receives the continuous state:

```text
(x, y)
```

and outputs four Q-values:

```text
[
    Q(Up),
    Q(Down),
    Q(Left),
    Q(Right)
]
```

### Network Architecture

```text
Input
  2
  │
  ▼
Linear(2 → 128)
  │
 ReLU
  │
  ▼
Linear(128 → 128)
  │
 ReLU
  │
  ▼
Linear(128 → 4)
  │
  ▼
Q-values
```

## DQN Components

### Experience Replay

Transitions are stored in a replay buffer:

```text
(state, action, reward, next_state, done)
```

Random mini-batches are sampled from the buffer during training.

### Target Network

A separate target network is used to calculate the Q-learning targets:

```text
target = reward + γ × max Q_target(next_state)
```

For terminal states, the future Q-value contribution is removed.

### Epsilon-Greedy Exploration

The agent initially explores the environment using random actions and gradually shifts towards exploiting the learned Q-values.

### Huber Loss

The Q-network is trained using the Smooth L1 / Huber loss between the predicted Q-value and the target Q-value.

## Results

The trained DQN can be evaluated by running the learned policy in the continuous maze.


---

# Key Concepts Demonstrated

- Reinforcement learning
- Q-learning
- Deep Q-learning
- Q-value estimation
- Epsilon-greedy exploration
- Stochastic environments
- Reward shaping
- Continuous state spaces
- Neural-network function approximation
- Experience replay
- Target networks
- Policy evaluation
- Training visualization

---
