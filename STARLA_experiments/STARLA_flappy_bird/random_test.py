import pickle

from ple import PLE
import numpy as np
from test_flappy_bird import TestFlappyBird
import time
import torch
import random
from copy import deepcopy


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(7, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, 2)

    def forward(self, input):
        output = self.fc1(input)
        output = torch.nn.functional.relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.relu(output)
        output = self.fc4(output)
        return output


def get_ple_state(t):
    return [
        t['player_vel'],
        t['player_y'] - t['next_pipe_bottom_y'],
        t['player_y'] - t['next_pipe_top_y'],
        t['next_pipe_dist_to_player'],
        t['player_y'] - t['next_next_pipe_bottom_y'],
        t['player_y'] - t['next_next_pipe_top_y'],
        t['next_next_pipe_dist_to_player'],
    ]


env1 = PLE(TestFlappyBird(), fps=30, display_screen=True)
model = torch.load('../result/flappy_bird/model/flappy_bird_model_repaired.pkl')

random.seed(0)
pipe1 = random.randint(25, 192)
pipe2 = random.randint(25, 192)
dist = random.randint(-120, -75)
vel = random.randint(-56, 10)
env1.game._init(pipe1, pipe2, dist, vel)

env2 = PLE(TestFlappyBird(), fps=30, display_screen=True)
env2.game._init(pipe1, pipe2, dist, vel)

for i in range(2):
    if i == 0:
        env = env1
    else:
        env = env2
    env.reset_game()
    s = get_ple_state(env.getGameState())
    episode_reward = 0
    step = 0
    action = 0
    while True:
        if env.game_over():
            break
        if episode_reward > 5:
            print(f'step:{step}')
            break
        # s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        # pred = model(s).cpu().detach().numpy()
        # action = np.argmax(pred)
        reward = env.act(env.getActionSet()[action])
        episode_reward += reward
        step += 1
        s = get_ple_state(env.getGameState())
        print(s)
        # time.sleep(0.04)

with open('./test.pickle', 'wb') as file:
    pickle.dump(test, file)
with open('./teststate.pickle', 'wb') as file:
    pickle.dump(teststate, file)

READ_DATA = True
if READ_DATA:
    with open('./test.pickle', 'rb') as file:
        test = pickle.load(file)
    with open('./teststate.pickle', 'rb') as file:
        teststate = pickle.load(file)
    with open('./uni1.pickle', 'rb') as file:
        uni1 = pickle.load(file)
    with open('./unique1.pickle', 'rb') as file:
        unique1 = pickle.load(file)
else:
    ee, qq = random_test_2(model, env2, 800_000)
    test, teststate = fix_testing(ee, qq, env2)
    d = 1
    unique1, uni1 = Abstract_classes(test, d, model)
unique5 = unique1