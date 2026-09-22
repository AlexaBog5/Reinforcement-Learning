# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy",
                     rewards_save_path="rewards.npy",
                     render=False):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    rewards = []

    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    for episode in range(no_episodes):
        state, _ = env.reset()

        state = tuple(state)
        total_reward = 0

        #! Step 2: Take actions in the environment until "Done" flag is triggered
        #! -------
        while True:
            #! Step 3: Define your Exploration vs. Exploitation
            #! -------
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit

            next_state, reward, done, _ = env.step(action)

            if render:
                env.render()

            next_state = tuple(next_state)
            total_reward += reward

            #! Step 4: Update the Q-values using the Q-value update rule
            #! -------
            q_table[state][action] = q_table[state][action] + alpha * \
                (reward + gamma *
                 np.max(q_table[next_state]) - q_table[state][action])

            state = next_state

            #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
            #! -------
            if done:
                break

        rewards.append(total_reward)
        #! Step 6: Perform epsilon decay
        #! -------
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}: Total Reward: {total_reward} Epsolon: {epsilon}")

    #! Step 7: Close the environment window
    #! -------
    env.close()
    print("Training finished.\n")

    #! Step 8: Save the trained Q-table
    #! -------
    np.save(q_table_save_path, q_table)
    np.save(rewards_save_path, rewards)

    print("Saved the Q-table.")


# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(goal_coordinates,
                      dog_state_coordinates,
                      puddle_state_coordinates,
                      actions=["Up", "Down", "Right", "Left"],
                      q_values_path="q_table.npy",
                      file_name="heatmap.jpg"):

    # Load the Q-table:
    # -----------------
    try:
        q_table = np.load(q_values_path)

        # Create subplots for each action:
        # --------------------------------
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            # Mask the goal state's Q-value for visualization:
            # ------------------------------------------------
            mask = np.zeros_like(heatmap_data, dtype=bool)

            mask[goal_coordinates] = True
            ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            
            for dog in dog_state_coordinates:
                mask[dog] = True
                ax.text(dog[1] + 0.5, dog[0] + 0.5, 'DOG', color='red',
                    ha='center', va='center', weight='bold', fontsize=14)

            for puddle in puddle_state_coordinates:
                mask[puddle] = True
                ax.text(puddle[1] + 0.5, puddle[0] + 0.5, 'P', color='blue',
                    ha='center', va='center', weight='bold', fontsize=14)
                
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
                        ax=ax, cbar=False, mask=mask, annot_kws={"size": 9}, vmin=-6, vmax=6)

            # Denote Goal and Hell states:
            # ----------------------------

            ax.set_title(f'Action: {action}')

        plt.tight_layout()
        plt.savefig(file_name)
        # plt.show()
        plt.clf()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

def visualize_rewards(window=20, rewards_save_path="rewards.npy", file_name="rewards.jpg"):
    rewards = np.load(rewards_save_path)
    plt.plot(rewards, label='Rewards', color='blue')

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
    plt.savefig(file_name)
    # plt.show()
    plt.clf()