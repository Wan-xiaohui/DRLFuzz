import torch
from test_pong import TestPong
from ple import PLE
import numpy as np
import random
import os

os.environ['SDL_VIDEODRIVER'] = "dummy"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

def test(net, arg):
    playerY = arg[0]
    playerVel = arg[1]
    ballX = arg[2]
    ballY = arg[3]
    ballVerX = arg[4]
    ballVerY = arg[5]
    p.reset_game()
    testPong._init(playerY, playerVel, ballX, ballY, ballVerX, ballVerY)
    t = p.getGameState()
    s = [
                t['player_y'],
                t['player_velocity'], 
                t['ball_x'],
                t['ball_y'], 
                t['ball_velocity_x'],
                t['ball_velocity_y'],
              ]
    score = 0
    while True:
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        pred = net(s).cpu().detach().numpy()
        a = np.argmax(pred)
        p.act(p.getActionSet()[a])
        t = p.getGameState()
        s_1 = [
                t['player_y'],
                t['player_velocity'], 
                t['ball_x'],
                t['ball_y'], 
                t['ball_velocity_x'],
                t['ball_velocity_y'],
              ]
        if s[4] <= 0 and s_1[4] >= 0:
            score += 1
        if testPong.game_over():
            break
        s = s_1
    return score

def verify(net, num):
    random.seed(2003511)
    scores = list()
    case = np.loadtxt(casePath)
    for c in case:
        score = test(net, (c[0], c[1], c[2], c[3], c[4], c[5]))
        scores.append(score)
    scores = np.array(scores)
    print([s for s in scores])
    print("Bad Cases mean:{} max{} min{} std{}".format(np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))
    random.seed(2003511)
    scores = list()
    for i in range(num):
        playerY = random.randint(30, 300)
        playerVel = random.randint(-15, 15)
        ballX = random.randint(350, 450)
        ballY = random.randint(50, 250)
        ballVerX = random.randint(-300, -250)
        ballVerY = random.randint(-200, 200)
        score = test(net, (playerY, playerVel, ballX, ballY, ballVerX, ballVerY))
        scores.append(score)
    scores = np.array(scores)
    print([s for s in scores])
    print("Random Cases mean:{} max{} min{} std{}".format(np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))

modelPath = "../result/pong/model/pong_model.pkl"
fixedModelPath= "../result/pong/model/pong_model_repaired.pkl"
casePath = "../result/pong/result_DRLFuzz.txt"
testPong = TestPong(width=64*8, height=48*8, MAX_SCORE=1)
p = PLE(testPong, fps=30, display_screen=False, force_fps=True)
p.init()
gamma = 0.9

if __name__ == '__main__':
    print("Before retraining")
    net = torch.load(modelPath).cuda().eval()
    verify(net, 10000)
    print("After retraining")
    net = torch.load(fixedModelPath).cuda().eval()
    verify(net, 10000)
    