import random

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