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
    s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
    while True:
        pred = net(s)
        label = net(s)
        a = torch.argmax(pred)
        reward = 0
        if abs(s[0]-s[2]) <  10 :
            reward +=  1
        else:
            reward += -2
        r = p.act(p.getActionSet()[a])
        if r == 0:
            reward += 1
        elif r > 0:
            reward += 5
        else:
            reward += -10
        t = p.getGameState()
        s = [
                t['player_x'],
                t['player_vel'],
                t['fruit_x'],
                t['fruit_y'],
              ]
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        if testCatcher.game_over():
            label[a] = reward
        else:
            q_1 = net(s)
            label[a] = reward + gamma * torch.max(q_1)
        stateList.append(s.cpu().detach().numpy())
        labelList.append(label.cpu().detach().numpy()) 
        if testCatcher.game_over() or p.score() > 10:
            break

modelPath = "../result/catcher/model/catcher_model.pkl"
savePath= "../result/catcher/model/catcher_model_repaired.pkl"
casePath = "../result/catcher/result_DRLFuzz.txt"
net = torch.load(modelPath).cuda().eval()
testCatcher = TestCatcher(width=64*8, height=64*8, init_lives=1)
p = PLE(testCatcher, fps=30, display_screen=False, force_fps=True)
p.init()
gamma = 0.9
random.seed(2003511)
stateList = list()
labelList = list()

if __name__ == '__main__':
    case = np.loadtxt(casePath)
    for c in case:
        getData((c[0], c[1], c[2], c[3]))
    repair(1e-5)
