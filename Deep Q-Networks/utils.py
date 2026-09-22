# Imports:
# --------
import imageio
import numpy as np
import torch
import random
from collections import deque
import torch.nn.functional as F


# Repla Buffer:
# -------------
class ReplayBuffer():
    def __init__(self, buffer_limit, device=None):
        self.buffer = deque(maxlen=buffer_limit)
        self.device = device if device is not None else \
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(s_lst), dtype=torch.float, device=self.device), \
               torch.tensor(a_lst, device=self.device), \
               torch.tensor(r_lst, device=self.device), \
               torch.tensor(np.array(s_prime_lst), dtype=torch.float, device=self.device), \
               torch.tensor(done_mask_lst, device=self.device)
    
    def size(self):
        return len(self.buffer)


# Train function:
# ---------------
def train(q_net, 
          q_target, 
          memory, 
          optimizer,
          batch_size,
          gamma):
    
    #! We sample from the same Replay Buffer n=10 times. 
    for _ in range(10):
        #! Monte Carlo sampling of a batch
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        #! Get the Q-values
        q_out = q_net(s)

        #! DQN update rule
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save GIF function:
# ----------------
def save_gif(env, dqn, max_steps, file_name="animation.gif", epsilon=0.1, fps=5):
    frames = []
    done = False

    s, _ = env.reset()
    env.render()
    frame = env.get_image()
    frames.append(frame)

    for _ in range(max_steps):
        rand_val = random.random()
        if rand_val < epsilon:
            action = random.randint(0, 3)
        else:
            action = dqn(torch.from_numpy(s).float()).argmax().item()
        s_prime, _, done, _, _ = env.step(action)
        env.render()
        frame = env.get_image()
        frames.append(frame)
        s = s_prime


        if done:
            break

    env.render()
    frame = env.get_image()
    frames.append(frame)
    
    env.close()
    # Save frames as a GIF
    imageio.mimsave(file_name, frames, fps=fps)
    print(f"Saved animation to {file_name}")