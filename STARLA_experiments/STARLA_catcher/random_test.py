import pickle

from ple import PLE
import numpy as np
from test_catcher import TestCatcher
import time
import torch
import random
from copy import deepcopy

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128 ,3)

    def forward(self, input):
        output = self.fc1(input)
        output = torch.nn.functional.relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.relu(output)
        output = self.fc3(output)
        return output


def get_ple_state(t):
    return [
                t['player_x'],
                t['player_vel'],
                t['fruit_x'],
                t['fruit_y'],
              ]


env1 = PLE(TestCatcher(width=64*8, height=64*8, init_lives=1), fps=30, display_screen=True)
model = torch.load('../result/catcher/model/catcher_model.pkl')

random.seed(10)
env1.init()

# env2 = PLE(TestCatcher(), fps=30, display_screen=True)
# env2.game._init(pipe1, pipe2, dist, vel)

for i in range(2):
    env = env1
    env.reset_game()
    env.game._init(11, 66, 266, 22)
    s = get_ple_state(env.getGameState())
    episode_reward = 0
    step = 0
    action = 0
    while True:
        if env.game_over():
            break
        if episode_reward > 10:
            print(f'step:{step}')
            break
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        pred = model(s).cpu().detach().numpy()
        action = np.argmax(pred)
        reward = env.act(env.getActionSet()[action])
        episode_reward += reward
        step += 1
        s = get_ple_state(env.getGameState())
        print(s)
        # time.sleep(0.03)