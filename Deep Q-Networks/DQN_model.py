# Import:
# -------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# Deep Q-Network:
# ---------------
class Qnet(nn.Module):
    def __init__(self, dim_actions, dim_states, device = None):
        super(Qnet, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(dim_states, 128, device=self.device)
        self.fc2 = nn.Linear(128, 128, device=self.device)
        self.fc3 = nn.Linear(128, dim_actions, device=self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, observation, epsilon):
        #! Exploration
        if random.random() < epsilon:
            return random.randint(0, self.fc3.out_features - 1) 
        
        #! Exploitation
        else:
            observation = observation.to(self.device)
            a = self.forward(observation)
            return a.argmax().item()
