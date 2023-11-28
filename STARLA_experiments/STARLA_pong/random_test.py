import copy
import pickle

from ple import PLE
import numpy as np
from test_pong import TestPong
import time
import torch
import random
from copy import deepcopy
random.seed(1)

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(6, 32)
        self.fc2 = torch.nn.Linear(32, 64)
        self.fc3 = torch.nn.Linear(64 ,64)
        self.fc4 = torch.nn.Linear(64 ,32)
        self.fc5 = torch.nn.Linear(32 ,3)

    def forward(self, input):
        output = self.fc1(input)
        output = torch.nn.functional.relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.relu(output)
        output = self.fc4(output)
        output = torch.nn.functional.relu(output)
        output = self.fc5(output)
        return output


def get_ple_state(t):
    return [
                t['player_y'],
                t['player_velocity'],
                t['ball_x'],
                t['ball_y'],
                t['ball_velocity_x'],
                t['ball_velocity_y'],
              ]


env1 = PLE(TestPong(width=64*8, height=48*8, MAX_SCORE=1), fps=30, display_screen=True)
model = torch.load('../result/pong/model/pong_model.pkl')

random.seed(0)
playerY = random.randint(30, 300)
playerVel = random.randint(-15, 15)
ballX = random.randint(350, 450)
ballY = random.randint(50, 250)
ballVerX = random.randint(-300, -250)
ballVerY = random.randint(-200, 200)
env1.init()
env1.game._init(playerY, playerVel, ballX, ballY, ballVerX, ballVerY)
env1.game.rewards['tick'] = 1

# env2 = PLE(TestCatcher(), fps=30, display_screen=True)
# env2.game._init(pipe1, pipe2, dist, vel)

for i in range(2):
    env = env1
    env.reset_game()
    s = get_ple_state(env.getGameState())
    episode_reward = 0
    step = 0
    action = 0
    s_1 = None
    while True:
        if env.game_over():
            break
        if step == 500:
            print(f'episode reward:{episode_reward}')
            break
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        pred = model(s).cpu().detach().numpy()
        action = np.argmax(pred)
        reward = env.act(env.getActionSet()[action])
        if reward != 0:
            print(reward)
        s = get_ple_state(env.getGameState())
        episode_reward += reward
        step += 1
        s_1 = copy.deepcopy(s)
        print(s)
        # time.sleep(0.02)