import pygame
import sys
import gymnasium as gym
import numpy as np
import os
import random

class MyEnv(gym.Env):
    def __init__(self, grid_size: int, random_initialization: bool, is_stochastic: bool, goal_coordinates: tuple) -> None:
        super().__init__()

        # Initialize environment parameters:
        self.random_initialization = random_initialization
        self.state = np.array([0,0])
        self.done = False
        self.info = {}
        self.reward = 0
        self.action = 3
        self.img_direction = 3
        self.wet_img_direction = 3

        self.is_stochastic = is_stochastic
        if self.is_stochastic:
            self.probability = 0.9
            self.is_wet = True
        else:
            self.probability = 1
            self.is_wet = False

        self.cell_size = 500 // grid_size
        if self.cell_size < 50:
            self.cell_size = 50
        self.grid_size = grid_size
        self.goal = np.array(goal_coordinates)
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=4, shape=(2,), dtype=np.int32)

        self.danger_states = []
        self.puddles_states = []

        # Load images:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_folder = os.path.join(script_dir, "imgs")
        self.agent_img = pygame.transform.scale(pygame.image.load(os.path.join(img_folder, "cat.png")), (self.cell_size, self.cell_size))
        self.goal_img = pygame.transform.scale(pygame.image.load(os.path.join(img_folder, "house.png")), (self.cell_size, self.cell_size))
        self.danger_img = pygame.transform.scale(pygame.image.load(os.path.join(img_folder, "dog.png")), (self.cell_size, self.cell_size))
        self.puddle_img = pygame.transform.scale(pygame.image.load(os.path.join(img_folder, "puddle.png")), (self.cell_size, self.cell_size))
        self.fight_img = pygame.transform.scale(pygame.image.load(os.path.join(img_folder, "fight.png")), (self.cell_size, self.cell_size))
        self.happy_end_img = pygame.transform.scale(pygame.image.load(os.path.join(img_folder, "happy_end.png")), (self.cell_size, self.cell_size))
        self.wet_cat_img = pygame.transform.scale(pygame.image.load(os.path.join(img_folder, "wet_cat.png")), (self.cell_size, self.cell_size))

        # display
        pygame.init()
        self.screen = pygame.display.set_mode((self.cell_size*self.grid_size, self.cell_size*(self.grid_size + 1)))

    def add_danger(self, coordinates: tuple) -> None:
        self.danger_states.append(np.array(coordinates))

    def add_puddle(self, coordinates: tuple) -> None:
        self.puddles_states.append(np.array(coordinates))

    def reset(self) -> None:
        if self.random_initialization:
            while True:
                self.state = np.array([np.random.randint(low=0, high=self.grid_size), np.random.randint(low=0, high=self.grid_size)])
                if not any(np.array_equal(self.state, danger) for danger in self.danger_states) and \
                    not any(np.array_equal(self.state, puddle) for puddle in self.puddles_states):
                    break
        else:
            self.state = np.array([0,0])
        self.done = False
        self.info = {}
        self.reward = 0
        self.action = 3

        if self.is_stochastic:
            self.probability = 0.9
            self.is_wet = True
        else:
            self.probability = 1
            self.is_wet = False

        self.info["Distance to goal"] = np.sqrt((self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2)
        self.info["Chosen action"] = None
        self.info["Real action"] = None

        # self.render()

        return self.state, self.info

    def step(self, action: int) -> tuple:
        if self.probability != 1:
            # Stochastic action selection:
            # Probability refers to the probability of taking the chosen action.
            actions = range(4)
            probabilities = [(1 - self.probability) / 3 if i != action else self.probability for i in actions]
            self.action = random.choices(actions, weights=probabilities, k=1)[0]
        else:
            self.action = action    
    
        self.info["Chosen action"] = action
        self.info["Real action"] = self.action

        ex_state = self.state.copy()
        match self.action:
            case 0: # up
                self.state[0] -= 1
            case 1: # down
                self.state[0] += 1
            case 2: # right
                self.state[1] += 1
            case 3: # left
                self.state[1] -= 1
            case _:
                pass

        # Check termination and give rewards
        if self.state[0] < 0 or self.state[0] > self.grid_size - 1 or \
            self.state[1] > self.grid_size - 1 or self.state[1] < 0: # border

            self.done = False
            self.reward = -0.1
            self.state = ex_state

        elif np.array_equal(self.state, self.goal): # goal
            self.done = True
            self.reward = 10

        elif any(np.array_equal(self.state, danger) for danger in self.danger_states): # danger
            self.done = True
            self.reward = -20

        elif any(np.array_equal(self.state, puddle) for puddle in self.puddles_states): # puddle
            self.done = False
            self.reward = -10
            if not self.is_wet and self.wet_img_direction != self.img_direction:
                self.wet_img_direction = self.img_direction
                self.wet_cat_img = pygame.transform.flip(self.wet_cat_img, True, False)
            self.is_wet = True

            if self.is_stochastic:
                self.probability -= 0.1
                if self.probability < 0.5:
                    self.probability = 0.5

        else: # empty
            self.done = False
            self.reward = -0.01

        self.info["Distance to goal"] = np.sqrt((self.state[0]-self.goal[0])**2 + (self.state[1]-self.goal[1])**2)
        

        return self.state, self.reward, self.done, self.info

    def render(self) -> None:
        # Code for closing the window
        for event in pygame.event.get():
            if event==pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Background:
        self.screen.fill((255, 240, 192))

        # Draw text with probability
        font = pygame.font.SysFont(None, 36)
        rect = pygame.Rect(0, 0, self.cell_size * self.grid_size, self.cell_size) 
        text_surface = font.render(f"p = {self.probability:.1f}", True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

        # Draw gridlines:
        for col in range(self.grid_size):
            for row in range(1, 1 + self.grid_size):
                grid = pygame.Rect(col*self.cell_size,
                            row*self.cell_size, 
                            self.cell_size, 
                            self.cell_size)
                pygame.draw.rect(self.screen, (199, 187, 146), grid, 1)

        # Draw goal:
        goal_pos = (self.goal[1] * self.cell_size, (self.goal[0] + 1) * self.cell_size)
        if self.done and np.array_equal(self.goal, self.state):
            self.screen.blit(self.happy_end_img, goal_pos)
        else:
            self.screen.blit(self.goal_img, goal_pos)

        # puddles 
        for puddle in self.puddles_states:
            if np.array_equal(puddle, self.state):
                continue
            puddle_pos = (puddle[1] * self.cell_size, (puddle[0] + 1) * self.cell_size)
            self.screen.blit(self.puddle_img, puddle_pos)

        # Add danger states:
        for danger in self.danger_states:
            danger_pos = (danger[1] * self.cell_size, (danger[0] + 1) * self.cell_size)
            if self.done and np.array_equal(danger, self.state):
                self.screen.blit(self.fight_img, danger_pos)
            else:
                self.screen.blit(self.danger_img, danger_pos)

        # Draw agent:
        agent_pos = (self.state[1] * self.cell_size, (self.state[0] + 1)* self.cell_size)

        if not self.done:
            img_attr = "wet_cat_img" if self.is_wet else "agent_img"
            direction_attr = "wet_img_direction" if self.is_wet else "img_direction"
            if self.action in [2,3] and self.action != getattr(self, direction_attr):
                setattr(self, direction_attr, self.action)
                setattr(self, img_attr, pygame.transform.flip(getattr(self, img_attr), True, False))
            self.screen.blit(getattr(self, img_attr), agent_pos)

        # pygame.time.wait(100)
        # pygame.time.wait(20)
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()


def create_env(grid_size: int,
               is_stochastic: bool,
               random_initialization: bool,
               goal_coordinates: tuple,
               dog_state_coordinates: list[tuple],
               puddle_state_coordinates: list[tuple]) -> MyEnv:
    # Create the environment:
    # -----------------------
    env = MyEnv(grid_size=grid_size, random_initialization=random_initialization, \
        is_stochastic=is_stochastic, goal_coordinates=goal_coordinates)

    for dog in dog_state_coordinates:
        env.add_danger(dog)

    for puddle in puddle_state_coordinates:
        env.add_puddle(puddle)

    return env
