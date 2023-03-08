import torch
from test_pong import TestPong
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

def test(arg):
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
    tmp = arg
    score = 0
    while True:
        dist = 0
        curr = ((int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]), int(s[5])))
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
        if testPong.game_over() or score > 10:
            break
        s = s_1
    return score

def mutator(arg, l):
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
    pred = net(s)
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
    playerY = playerY + grad[0]*l
    playerVel = playerVel + grad[1]*l
    ballX = ballX + grad[2]*l
    ballY = ballY + grad[3]*l
    ballVerX = ballVerX + grad[4]*l
    ballVerY = ballVerY + grad[5]*l
    playerY = max(min(playerY, 300), 30)
    playerVel = max(min(playerVel, 15), -15)
    ballX = max(min(ballX, 450), 350)
    ballY = max(min(ballY, 250), 50)
    ballVerX = max(min(ballVerX, -250), -300)
    ballVerY = max(min(ballVerY, 200), -200)
    return [playerY, playerVel, ballX, ballY, ballVerX, ballVerY]

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
            playerY = random.randint(30, 300)
            playerVel = random.randint(-15, 15)
            ballX = random.randint(350, 450)
            ballY = random.randint(50, 250)
            ballVerX = random.randint(-300, -250)
            ballVerY = random.randint(-200, 200)
            count += 1
            if count == 10000:
                delta *= 0.9
            if getDistance((playerY, playerVel, ballX, ballY, ballVerX, ballVerY)) > delta:
                allStates.add((playerY, playerVel, ballX, ballY, ballVerX, ballVerY))
                break
    else:
        playerY = random.randint(30, 300)
        playerVel = random.randint(-15, 15)
        ballX = random.randint(350, 450)
        ballY = random.randint(50, 250)
        ballVerX = random.randint(-300, -250)
        ballVerY = random.randint(-200, 200)
    return [playerY, playerVel, ballX, ballY, ballVerX, ballVerY]

def getDistance(arg):
    if kdTree is None:
        return np.inf
    else:
        dist, _ = kdTree.query(np.array(list(arg)))
        return dist

testPong = TestPong(width=64*8, height=48*8, MAX_SCORE=1)
path = "../result/pong/model/pong_model.pkl"
net = torch.load(path).cuda().eval()
savePath = "../result/pong/result_DRLFuzz.txt"
p = PLE(testPong, fps=30, display_screen=False, force_fps=True)
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
    result = DRLFuzz(100,1000, 10, 0.1, 1, True)

