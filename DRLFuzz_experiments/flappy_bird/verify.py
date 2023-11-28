import torch
from test_flappy_bird import TestFlappyBird
from ple import PLE
import numpy as np
import random
import os

os.environ['SDL_VIDEODRIVER'] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(7, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128 ,64)
        self.fc4 = torch.nn.Linear(64 ,2)

    def forward(self, input):
        output = self.fc1(input)
        output = torch.nn.functional.relu(output)
        output = self.fc2(output)
        output = torch.nn.functional.relu(output)
        output = self.fc3(output)
        output = torch.nn.functional.relu(output)
        output = self.fc4(output)
        return output

def test(net, arg):
    pipe1 = arg[0]
    pipe2 = arg[1]
    dist = arg[2]
    vel = arg[3]
    p.reset_game()
    testFlappybird._init(pipe1, pipe2, dist, vel)
    t = p.getGameState()
    s = [
                t['player_vel'],
                t['player_y']-t['next_pipe_bottom_y'], 
                t['player_y']-t['next_pipe_top_y'],
                t['next_pipe_dist_to_player'],
                t['player_y']-t['next_next_pipe_bottom_y'], 
                t['player_y']-t['next_next_pipe_top_y'],
                t['next_next_pipe_dist_to_player'],
              ]
    while True:
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        pred = net(s).cpu().detach().numpy()
        a = np.argmax(pred)
        p.act(p.getActionSet()[a])
        if testFlappybird.game_over():
            break
        t = p.getGameState()
        s = [
                t['player_vel'],
                t['player_y']-t['next_pipe_bottom_y'], 
                t['player_y']-t['next_pipe_top_y'],
                t['next_pipe_dist_to_player'],
                t['player_y']-t['next_next_pipe_bottom_y'], 
                t['player_y']-t['next_next_pipe_top_y'],
                t['next_next_pipe_dist_to_player'],
              ]
    return p.score()+5

def verify(net, num):
    random.seed(2003511)
    scores = list()
    case = np.loadtxt(casePath)
    for c in case:
        score = test(net, (c[0], c[1], c[2], c[3]))
        scores.append(score)
    scores = np.array(scores)
    print([s for s in scores])
    print("Bad Cases mean:{} max{} min{} std{}".format(np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))
    random.seed(2003511)
    scores = list()
    for i in range(num):
        pipe1 = random.randint(25, 192)
        pipe2 = random.randint(25, 192)
        dist = random.randint(-120, -75)
        vel = random.randint(-56, 10)
        score = test(net, (pipe1, pipe2, dist, vel))
        scores.append(score)
    scores = np.array(scores)
    print([s for s in scores])
    print("Random Cases mean:{} max{} min{} std{}".format(np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))

modelPath = "../result/flappy_bird/model/flappy_bird_model.pkl"
fixedModelPath= "../result/flappy_bird/model/flappy_bird_model_repaired.pkl"
casePath = "../result/flappy_bird/result_DRLFuzz.txt"
testFlappybird = TestFlappyBird()
p = PLE(testFlappybird, fps=30, display_screen=False, force_fps=True)
p.init()
gamma = 0.9

if __name__ == '__main__':
    print("Before retraining")
    net = torch.load(modelPath).cuda().eval()
    verify(net, 10000)
    print("After retraining")
    net = torch.load(fixedModelPath).cuda().eval()
    verify(net, 10000)
    