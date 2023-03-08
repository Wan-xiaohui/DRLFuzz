import torch
from test_catcher import TestCatcher
from ple import PLE
import numpy as np
import random
import os
from scipy import spatial
import sys

sys.setrecursionlimit(10000000)
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

def test(arg):
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
    tmp = arg
    while True:
        dist = 0
        curr = ((int(s[1]), int(s[0]+51), int(s[2]), int(s[3])))
        for i in range(len(tmp)):
            dist += (tmp[i] - curr[i])**2
        dist = dist**0.5
        if dist > innerDelta:
            tmp = curr
            if getDistance(curr) > delta:
                allStates.add(curr)
        s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
        pred = net(s).cpu().detach().numpy()
        a = np.argmax(pred)
        p.act(p.getActionSet()[a])
        if testCatcher.game_over() or p.score() > 10:
            break
        t = p.getGameState()
        s = [
                t['player_x'],
                t['player_vel'],
                t['fruit_x'],
                t['fruit_y'],
              ]
    return p.score()+6

def mutator(arg, l):
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
    pred = net(s)
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
    s_1 = [
                t['player_x'],
                t['player_vel'],
                t['fruit_x'],
                t['fruit_y'],
              ]
    s_1 = torch.tensor(s_1, dtype=torch.float32, requires_grad=False).cuda()
    q_1 = net(s_1)
    label = pred.clone().detach()
    label[a] = r + 0.9*torch.max(q_1)

    s.requires_grad = True
    net.zero_grad()
    criterion = torch.nn.MSELoss()
    loss = criterion(net(s), label)
    loss.backward()
    grad = s.grad.cpu().numpy()
    vel = vel+grad[1]*l
    playerX = playerX + grad[0]*l
    fruitX = fruitX + grad[2]*l
    fruitY = fruitY + grad[3]*l
    vel = max(min(vel, 30), -30)
    playerX = max(min(playerX, 461), 51)
    fruitX = max(min(fruitX, 448), 64)
    fruitY = max(min(fruitY, 50), -150)
    return [vel, playerX, fruitX, fruitY]

def DRLFuzz(num, n, l, alpha, theta, coverage):
    global kdTree
    statePool = list()
    score = list()
    resultPool = set()
    for _ in range(num):
        s = randFun(coverage)
        statePool.append(s)
        score.append(0)
    for k in range(n):
        for i in range(num):
            score[i] = test(statePool[i])
            if (score[i] < theta):
                tmp = [int(x) for x in statePool[i]]
                if tuple(tmp) not in resultPool:
                    with open(savePath, 'a') as f:
                        for j in range(len(tmp)):
                            f.write(str(tmp[j])+' ')
                        f.write('\n')
                resultPool.add(tuple(tmp))
        kdTree = spatial.KDTree(data=np.array(list(allStates)), leafsize=10000)
        print("iteration {} failed cases num:{}".format(k+1, len(resultPool)))
        resultNum.append(len(resultPool))
        idx = sorted(range(len(score)), key=lambda x: score[x])
        for i in range(num):
            if i < int(num*alpha):
                st = mutator(statePool[idx[i]], l)
                if st != statePool[idx[i]]:
                    statePool[idx[i]] = st
                else:
                    statePool[idx[i]] = randFun(coverage)
            else:
                statePool[idx[i]] = randFun(coverage)
    return resultPool
    
def randFun(coverage):
    if coverage:
        global delta
        count = 0
        while True:
            vel = random.randint(-30, 30)
            playerX = random.randint(51, 461)
            fruitX = random.randint(64, 448)
            fruitY = random.randint(-150, 50)
            count += 1
            if count == 10000:
                delta *= 0.9
            if getDistance((vel, playerX, fruitX, fruitY)) > delta:
                allStates.add((vel, playerX, fruitX, fruitY))
                break
    else:
        vel = random.randint(-30, 30)
        playerX = random.randint(51, 461)
        fruitX = random.randint(64, 448)
        fruitY = random.randint(-150, 50)
    return [vel, playerX, fruitX, fruitY]

def getDistance(arg):
    if kdTree is None:
        return np.inf
    else:
        dist, _ = kdTree.query(np.array(list(arg)))
        return dist

testCatcher = TestCatcher(width=64*8, height=64*8, init_lives=1)
path = "../result/catcher/model/catcher_model.pkl"
net = torch.load(path).cuda().eval()
savePath = "../result/catcher/result_DRLFuzz.txt"
p = PLE(testCatcher, fps=30, display_screen=False, force_fps=True)
p.init()
resultNum = list()
allStates = set()
random.seed(2003511)
kdTree = None
delta = 20
innerDelta = 20

if __name__ == '__main__':
    if (os.path.exists(savePath)) :
        os.remove(savePath)
    result = DRLFuzz(100, 1000, 10, 0.1, 1, True)
