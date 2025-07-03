# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import os
import random
import torch
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from env import *
from utils import save_gif

# User definitions:
# -----------------
train_dqn = True
test_dqn = True
render = True
need_save_gif = True


#! Define env attributes (environment specific)
dim_actions = 4
dim_states = 2


# Hyperparameters:
# ----------------
learning_rate = 0.005
gamma = 0.9
buffer_limit = 50_000
batch_size = 32
num_episodes = 3_000
num_episodes_before_min_epsilon = 2_000
max_steps = 500

folder_path = "runs"
run_postfix = "final"

script_dir = os.path.dirname(os.path.abspath(__file__))
run_folder = os.path.join(script_dir, folder_path)
os.makedirs(run_folder, exist_ok=True)

dqn_weights_path = os.path.join(run_folder, f"dqn_{run_postfix}.pth")
reward_curve_path = os.path.join(run_folder, f"training_curve_{run_postfix}.jpg")

# Main:
# -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if train_dqn:
    if render:
        env = ContinuousMazeEnv(render_mode="human")
    else:
        env = ContinuousMazeEnv(render_mode="no_render")

    #! Initialize the Q Net and the Q Target Net
    q_net = Qnet(dim_actions=dim_actions, 
                 dim_states=dim_states).to(device)
    q_target = Qnet(dim_actions=dim_actions, 
                    dim_states=dim_states).to(device)
    
    q_target.load_state_dict(q_net.state_dict())

    #! Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit, device=device)

    print_interval = 10
    episode_reward = 0.0
    optimizer = optim.Adam(q_net.parameters(),
                           lr=learning_rate)

    rewards = []
    reached_goal = []


    for n_epi in range(num_episodes):
        #! Epsilon decay 
        epsilon = max(0.1, 1. - (n_epi/(num_episodes_before_min_epsilon - 1)) * 0.9)  

        s, _ = env.reset()
        done = False

        #! Define maximum steps per episode
        for _ in range(max_steps):
            #! Choose an action (Exploration vs. Exploitation)
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _, info = env.step(a)

            done_mask = 0.0 if done else 1.0

            #! Save the trajectories
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            episode_reward += r

            if done:
                break

        if memory.size() > 2000:
            train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}, reached_goal : {info["reached_goal"]}")

        if epsilon == 0.1:
            reached_goal.append(info["reached_goal"])

        rewards.append(episode_reward)
        episode_reward = 0.0


        #! Define a stopping condition for the game:
        if len(reached_goal) >= 100 and all(reached_goal[-100:]):
            print(f"Condition satisfied! Episode: {n_epi}")
            break

    env.close()

    #! Save the trained Q-net
    torch.save(q_net.state_dict(), dqn_weights_path)

    #! Plot the training curve
    plt.plot(rewards, label='Rewards', color='blue')

    #! Calculate and plot the moving average
    window = 15
    summation = sum(rewards[:window + 1])
    num = window + 1
    moving_average = [summation / num]
    for i in range(1, len(rewards)):
        if i + window < len(rewards):
            num += 1
            summation += rewards[i + window]
        if i - window >= 0:
            num -= 1
            summation -= rewards[i - window]

        moving_average.append(summation / num)
    plt.plot(moving_average, label='Moving Average', color='red', linewidth=3)

    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title("Rewards per Epoch")
    plt.legend()
    plt.savefig(reward_curve_path)
    plt.show()


if test_dqn or need_save_gif:
    dqn = Qnet(dim_actions=dim_actions, 
               dim_states=dim_states)
    dqn.load_state_dict(torch.load(dqn_weights_path))

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    if render:
        env = ContinuousMazeEnv(render_mode="human")
    else:
        env = ContinuousMazeEnv(render_mode="no_render")

    epsilon = 0.1  
    for _ in range(20):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            rand_val = random.random()
            if rand_val < epsilon:
                action = random.randint(0, 3)
            else:
                action = dqn(torch.from_numpy(s).float()).argmax().item()
            s_prime, reward, done, _, _ = env.step(action)
            env.render()
            s = s_prime

            episode_reward += reward

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()

# Save GIF:
if need_save_gif:
    env = ContinuousMazeEnv(render_mode="human")

    save_gif(env=env, dqn=dqn,
              max_steps=max_steps,
              file_name=os.path.join(run_folder, f"policy_{run_postfix}.gif"),
              epsilon=0.0,
              fps=5)