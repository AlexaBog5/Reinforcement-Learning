import random
import imageio
import numpy as np

def generate_unique_tuples(n, x_range, y_range, forbidden, random_seed):
    random.seed(random_seed)
    all_possible = [
        (y, x)
        for x in range(x_range + 1)
        for y in range(y_range + 1)
        if (y, x) not in forbidden
    ]

    if len(all_possible) < n:
        raise ValueError("Not enough valid tuples available to generate the requested number.")

    return random.sample(all_possible, n)

def save_gif(env, file_name="animation.gif", q_values_path="q_table.npy", fps=5):
    frames = []
    state, _ = env.reset()
    done = False
    q_table = np.load(q_values_path)

    while not done:
        env.render()
        frame = env.get_image()
        frames.append(frame)
        action = np.argmax(q_table[tuple(state)])
        state, _, done, _ = env.step(action)

    env.render()
    frame = env.get_image()
    frames.append(frame)
    
    env.close()
    # Save frames as a GIF
    imageio.mimsave(file_name, frames, fps=fps)
    print(f"Saved animation to {file_name}")