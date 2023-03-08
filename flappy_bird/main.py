import torch
from test_flappy_bird import TestFlappyBird
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

def test(arg):
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
    tmp = arg
    while True:
        dist = 0
        curr = ((int(t['player_y']-s[2]), int(t['player_y']-s[5]), int(s[3]-309), int(s[0])))
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
        if testFlappybird.game_over() or p.score() > 10:
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

def mutator(arg, l):
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
    s = torch.tensor(s, dtype=torch.float32, requires_grad=False).cuda()
    pred = net(s)
    a = torch.argmax(pred)
    reward = 0
    if s[1] <  0 and s[2] > 0:
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
                t['player_vel'],
                t['player_y']-t['next_pipe_bottom_y'], 
                t['player_y']-t['next_pipe_top_y'],
                t['next_pipe_dist_to_player'],
                t['player_y']-t['next_next_pipe_bottom_y'], 
                t['player_y']-t['next_next_pipe_top_y'],
                t['next_next_pipe_dist_to_player'],
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
    pipe1 = pipe1-grad[1]*l
    pipe2 = pipe2-grad[4]*l
    dist = dist+grad[3]*l
    vel = vel+grad[0]*l
    pipe1 = max(min(pipe1, 192), 25)
    pipe2 = max(min(pipe2, 192), 25)
    dist = max(min(dist, -75), -120)
    vel = max(min(vel, 10), -56)
    return [pipe1, pipe2, dist, vel]

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
            pipe1 = random.randint(25, 192)
            pipe2 = random.randint(25, 192)
            dist = random.randint(-120, -75)
            vel = random.randint(-56, 10)
            count += 1
            if count == 10000:
                delta *= 0.9
            if getDistance((pipe1, pipe2, dist, vel)) > delta:
                allStates.add((pipe1, pipe2, dist, vel))
                break
    else:
        pipe1 = random.randint(25, 192)
        pipe2 = random.randint(25, 192)
        dist = random.randint(-120, -75)
        vel = random.randint(-56, 10)
    return [pipe1, pipe2, dist, vel]

def getDistance(arg):
    if kdTree is None:
        return np.inf
    else:
        dist, _ = kdTree.query(np.array(list(arg)))
        return dist

testFlappybird = TestFlappyBird()
path = "../result/flappy_bird/model/flappy_bird_model.pkl"
net = torch.load(path).cuda().eval()
savePath = "../result/flappy_bird/result_DRLFuzz.txt"
p = PLE(testFlappybird, fps=30, display_screen=False, force_fps=True)
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
    result = DRLFuzz(100, 1000, 10, 0.1, 2, True)
