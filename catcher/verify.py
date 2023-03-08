import torch
from test_catcher import TestCatcher
from ple import PLE
import numpy as np
import random
import os

os.environ['SDL_VIDEODRIVER'] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

def test(net, arg):
    vel = arg[0]
    playerX = arg[1]
    fruitX = arg[2]
    fruitY = arg[3]
    p.reset_game()
    testCatcher._init(vel, playerX, fruitX, fruitY)
    t = p.getGameState()
    s = [
                t['player_x'],
                t['player_vel'],
                t['fruit_x'],
                t['fruit_y'],
              ]
    while True:
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        pred = net(s).cpu().detach().numpy()
        a = np.argmax(pred)
        p.act(p.getActionSet()[a])
        if testCatcher.game_over():
            break
        t = p.getGameState()
        s = [
                t['player_x'],
                t['player_vel'],
                t['fruit_x'],
                t['fruit_y'],
              ]
    return p.score()+6

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
        vel = random.randint(-30, 30)
        playerX = random.randint(51, 461)
        fruitX = random.randint(64, 448)
        fruitY = random.randint(-150, 50)
        score = test(net, (vel, playerX, fruitX, fruitY))
        scores.append(score)
    scores = np.array(scores)
    print([s for s in scores])
    print("Random Cases mean:{} max{} min{} std{}".format(np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))

modelPath = "../result/catcher/model/catcher_model.pkl"
fixedModelPath= "../result/catcher/model/catcher_model_repaired.pkl"
casePath = "../result/catcher/result_DRLFuzz.txt"
testCatcher = TestCatcher(width=64*8, height=64*8, init_lives=1)
p = PLE(testCatcher, fps=30, display_screen=False, force_fps=True)
p.init()
gamma = 0.9

if __name__ == '__main__':
    print("Before retraining")
    net = torch.load(modelPath).cuda().eval()
    verify(net, 10000)
    print("After retraining")
    net = torch.load(fixedModelPath).cuda().eval()
    verify(net, 10000)
    