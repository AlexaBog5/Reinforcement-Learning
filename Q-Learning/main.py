# Imports:
# --------
# from project import create_env
from env import create_env
from Q_learning import *
import os
from utils import *

# User definitions:
# -----------------
train = False
visualize_results = False
need_save_gif = True

learning_rate = 0.02  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.9995  # Decay rate for exploration
no_episodes = 10000  # Number of episodes

random_initialization = False
is_stochastic = True
folder_path = "runs"
run_postfix = "large_stochastic_rand_init_lr002_g099_decay09995_episodes10000"

script_dir = os.path.dirname(os.path.abspath(__file__))
run_folder = os.path.join(script_dir, folder_path)
os.makedirs(run_folder, exist_ok=True)

q_table_path = os.path.join(run_folder, f"q_table_{run_postfix}.npy")
rewards_path = os.path.join(run_folder, f"rewards_{run_postfix}.npy")

# Define the grid size and coordinates:
# -----------------------------------
# small grid example:
# grid_size = 5
# goal_coordinates = (4, 4)
# dog_state_coordinates = [(0, 2), (3, 1)]
# puddle_state_coordinates = [(1, 0), (2, 4), (3, 2)]

# large grid example:
grid_size = 10
goal_coordinates = (9, 9)
dog_state_coordinates = generate_unique_tuples(15, 9, 9, [goal_coordinates, [0,0]], 3)
puddle_state_coordinates = generate_unique_tuples(10, 9, 9, [goal_coordinates, [0,0]] + dog_state_coordinates, 5)

# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(grid_size=grid_size,
                     is_stochastic=is_stochastic,
                     random_initialization=random_initialization,
                     goal_coordinates=goal_coordinates,
                     dog_state_coordinates=dog_state_coordinates,
                     puddle_state_coordinates=puddle_state_coordinates)

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma,
                     q_table_save_path=q_table_path,
                     rewards_save_path=rewards_path)

if visualize_results:

    # Visualize and save the Q-table:
    # ----------------------
    visualize_q_table(dog_state_coordinates=dog_state_coordinates,
                      puddle_state_coordinates=puddle_state_coordinates,
                      goal_coordinates=goal_coordinates,
                      q_values_path=q_table_path,
                      file_name=os.path.join(run_folder, f"heatmap_{run_postfix}.jpg"))
    # Visualize and save the rewards:
    # ----------------------
    visualize_rewards(rewards_save_path=rewards_path,
                      file_name=os.path.join(run_folder, f"rewards_{run_postfix}.jpg"))

if need_save_gif:
    # Create a GIF from the saved images:
    # -----------------------------------
    env = create_env(grid_size=grid_size,
                    is_stochastic=is_stochastic,
                    random_initialization=random_initialization,
                    goal_coordinates=goal_coordinates,
                    dog_state_coordinates=dog_state_coordinates,
                    puddle_state_coordinates=puddle_state_coordinates)
    save_gif(env=env,
              file_name=os.path.join(run_folder, f"policy_{run_postfix}.gif"),
              q_values_path=q_table_path,
              fps=5)