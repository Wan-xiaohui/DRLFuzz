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

def repair(lr):
    global stateList
    global labelList
    batchSize = 1024
    n = int((len(stateList)-1) / batchSize) + 1
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    optimizer.zero_grad()
    for i in range(n):
        data = torch.tensor(stateList[i*batchSize: min(i*batchSize+batchSize, len(stateList))], dtype=torch.float32).cuda()
        label = torch.tensor(labelList[i*batchSize: min(i*batchSize+batchSize, len(labelList))], dtype=torch.float32).cuda()
        pred = net(data)
        loss = criterion(pred, label)
        loss.backward()
    optimizer.step()
    torch.save(net, savePath)

def getData(arg):
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
    s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
    score = 0
    while True:
        pred = net(s)
        label = net(s)
        a = torch.argmax(pred)
        reward = 0
        if abs(s[0]-s[3]) <  25 :
            reward +=  1
        else:
            reward += -2
        r = p.act(p.getActionSet()[a])
        if r == 0:
            reward += 1
        if r < 0:
            reward += -10
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
            reward += 2
            score += 1
        s = s_1
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        if testPong.game_over():
            label[a] = reward
        else:
            q_1 = net(s)
            label[a] = reward + gamma * torch.max(q_1)
        stateList.append(s.cpu().detach().numpy())
        labelList.append(label.cpu().detach().numpy()) 
        if testPong.game_over() or score > 10:
            break

modelPath = "../result/pong/model/pong_model.pkl"
savePath= "../result/pong/model/pong_model_repaired.pkl"
casePath = "../result/pong/result_DRLFuzz.txt"
net = torch.load(modelPath).cuda().eval()
testPong = TestPong(width=64*8, height=48*8, MAX_SCORE=1)
p = PLE(testPong, fps=30, display_screen=False, force_fps=True)
p.init()
gamma = 0.9
random.seed(2003511)
stateList = list()
labelList = list()

if __name__ == '__main__':
    case = np.loadtxt(casePath)
    for c in case:
        getData((c[0], c[1], c[2], c[3], c[4], c[5]))
    repair(1e-3)
