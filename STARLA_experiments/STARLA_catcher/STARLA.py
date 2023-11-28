from test_catcher import TestCatcher
import torch
import numpy as np
from ple import PLE
import math
from random import seed
import random
from datetime import datetime
import pickle
import sklearn
import numpy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.utils import resample
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import f1_score
from sklearn import impute
import statistics
from scipy import stats
from copy import deepcopy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from math import ceil
import copy
import sys
import os
from sklearn.metrics import jaccard_score
import time
import multiprocessing
from pymoo.algorithms.moo.nsga2 import calc_crowding_distance

import subprocess
import logging
from csv import reader

DISPLAY_SCREEN = False
random.seed(1)
DD = 1


class PLEStoreWrapper:
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env: PLE):
        self.max_steps = 500
        # Counter of steps per episode
        self.current_step = 0
        self.mem = []
        self.TotalReward = 0.0
        self.env = env
        self.first_state = None
        self.first_obs = 0
        self.prev_obs = 0
        self.states_list = []
        self.env.init()

    def reset(self):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        self.env.reset_game()
        vel = random.randint(-30, 30)
        playerX = random.randint(51, 461)
        fruitX = random.randint(64, 448)
        fruitY = random.randint(-150, 50)
        self.set_state([vel, playerX, fruitX, fruitY])
        obs = self.get_ple_state()
        self.TotalReward = 0.0
        self.first_obs = obs
        return obs

    def step(self, action):
        """
        In this function we store the initial state as well as the memory of the agent
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
        """
        if self.current_step == 0:  # store initial state
            self.prev_obs = self.first_obs
            if self.first_state is None:
                assert False, 'Please set first_state'
            self.states_list.append(self.first_state)
        self.current_step += 1
        reward = self.env.act(self.env.getActionSet()[action])
        if reward == 0:
            reward = 1
        obs = self.get_ple_state()
        done = self.env.game_over()
        self.TotalReward += reward
        self.mem.append(tuple((self.prev_obs, action)))
        self.prev_obs = obs
        if self.current_step >= self.max_steps:
            done = True
            # Update the info dict to signal that the limit was exceeded
        if done:
            self.mem.append(tuple(('done', self.TotalReward)))
        info = {'mem': self.mem, 'state': self.states_list}
        return obs, reward, done, info

    def set_state(self, state):
        """
        :param state: initial state of the episode
        :return: environment is updated and observations is returned
        """
        self.first_state = state
        self.env.game._init(state[0], state[1], state[2], state[3])
        obs = self.get_ple_state()
        self.current_step = 0
        self.TotalReward = 0.0
        self.first_obs = obs
        return obs

    def get_ple_state(self):
        t = self.env.getGameState()
        s = [
                t['player_x'],
                t['player_vel'],
                t['fruit_x'],
                t['fruit_y'],
              ]
        return np.array(s)


def is_fail_state(state, epsilon=0):
    state = np.array(state)
    if state.ndim == 1:
        if state[3] > 460:
            if state[2] > state[0] + 55 or state[2] < state[0] - 55:
                return True
        return False
    else:
        idx = state[:, 3] > 460
        tmp = state[idx][:, 2] - state[idx][:, 0]
        if (tmp > 55).any() or (tmp < -55).any():
            return True
        return False


class TorchModel():
    def __init__(self, torch_net: torch.nn.Module):
        self.torch_net = torch_net

    def abstract_state(self, state1, d):
        if type(state1) == str:
            if state1 == 'done':
                return 'end'
        state1 = torch.tensor(state1, dtype=torch.float32, requires_grad=False).cuda()
        q_value = self.torch_net(state1).cpu().detach().numpy()
        if q_value.ndim == 1:
            return tuple(np.ceil(q_value / d))
        else:
            return [tuple(i) for i in np.ceil(q_value / d)]

    def predict(self, obs, deterministic=True):
        obs = torch.tensor(obs, dtype=torch.float32, requires_grad=False).cuda()
        q_value = self.torch_net(obs).cpu().detach().numpy()
        if deterministic:
            return np.argmax(q_value)
        else:
            return np.random.choice([0, 1], p=q_value / q_value.sum())

    def action_probability(self, state):
        state = torch.tensor(state, dtype=torch.float32, requires_grad=False).cuda()
        q_value = self.torch_net(state).cpu().detach().numpy()
        return q_value / q_value.sum()


def proportional_sampling_whitout_replacement(index, size):
    s = 0
    s = sum(np.array(index))
    p = [ind / s for ind in index]
    samples = np.random.choice(index, size=size, replace=False, p=p)
    return samples


def population_sample(episodes, ind, pop_size, random_test_size, threshold, functional_fault_size, reward_fault_size):
    """
    This function is meant to sample episodes from training after that you need to add test episodes using random_test
    Set the parameters as you want but be careful the input episodes for this function is the memory of the agent and each step has seperate index
    this function returs the final steps of the selected function then you need to extract that episodes from the input memore that is called 'episodes'
    use the episodes extract function ...

    samples n episodes from training n1 functinal faults and n2 reward faults
    reward faults are episodes with reward bellow the thresthreshold
    from random test samples M episodes m1 random episode and
    m2 episodes with sudden reward change we dont have a sudden reward change in this example
    """
    epsilon = 5
    index = []
    functional_fault = []
    reward_fault = []
    start_states = []
    ind = np.where(np.array(episodes) == ('done',))
    index = ind[0]
    print(len(ind[0]), 'episodes from training')
    population = []
    for i in index:
        _, r = episodes[i]
        if is_fail_state(episodes[i-1][0]):
            functional_fault.append(i)
            print('function fault')
        if r < threshold:
            reward_fault.append(i)
            print('reward fault')
    if len(functional_fault) < functional_fault_size:
        print('functional faults size is', len(functional_fault), ' and its less than desired number')
        population += functional_fault
        print('sampling more random episodes instead ...!')
    if len(functional_fault) == functional_fault_size:
        population += functional_fault
    if len(functional_fault) > functional_fault_size:
        # proportianl_sample_whitout_replacement()
        sam1 = proportional_sampling_whitout_replacement(functional_fault, functional_fault_size)
        population += sam1
    if len(reward_fault) < reward_fault_size:
        print('reward faults size is', len(reward_fault), ' and its less than desired number')
        population += reward_fault
        print('sampling more random episodes instead ...!')
    if len(reward_fault) == reward_fault_size:
        population += reward_fault
    if len(reward_fault) > reward_fault_size:
        # proportional sampling
        sam2 = proportional_sampling_whitout_replacement(reward_fault, reward_fault_size)
        population += list(sam2)
    r_size = pop_size - len(population)
    # random_test(model,env,r_size)
    print(len(reward_fault))
    # population += reward_fault
    return population, r_size


def episode_extract(sampled_index, episodes):
    epis = []
    for i in sampled_index:
        l = int(episodes[i][1])
        slice1 = episodes[(i - l):(i + 1)]
        epis += slice1
    return epis


def fitness_reward(episode):
    """
    here the reward could be calculated as the lengh of the episode; Since the
    reward of the cartpole is defined based on the number of steps without falling
    last part of the episode contains the signal of ('done',reward)
    """
    return len(episode) - 1


def fitness_confidence(episode, model, mode):
    """
    confidence level is define as differences between the highest and
    second highest action probabilities of selecting actions OR
    the ratio between the highest and lowest/second highest action probability
    :param `mode`: r for ration and m for differences
    :param `model`: is the RL agent
    :param `episode`: is the episode values or sequence from the rl
    """
    cl = 0.0
    for i in range(len(episode)):
        if i == (len(episode) - 1):
            if episode[i][0] == 'done':
                return (cl / episode[i][1])
            else:
                assert False, "last state is not done , reward"
        else:
            prob = model.action_probability(episode[i][0])
            high1 = prob.argmax()
            first = prob[high1]
            temp = prob
            temp[high1] = 0.0
            high2 = temp.argmax()
            second = prob[high2]
            if mode == 'r':
                cl += (first / second)
                # In the next version this will be updated to a normalized ratio to avoid having large values
            if mode == 'm':
                cl += (first - second)  # To_Do: first - second / first +second this one is better
    print("WARNING nothing returned", episode)


def fitness_reward_probability(ml, binary_episode):
    """
    This function returns the third fitness funciton that is ment to guide the search toward
    the episodes with a higher probability of a reward fault and as we have a minimizing
    optimization funciton in MOSA we neeed to change this functionwe can either go with the
    negation of the probability of the reward fault = 1-probability of the reward fault
    that is equal to the probability of the bein a non-faulty episode
    :param `ml`: RF_FF_1rep for functional fault
    :param `binary episode`: episodes decodeed as having abstract states
    """
    # return -(ml.predict_proba(episode)[0][1])
    return ml.predict_proba(binary_episode)[0][0]


def fitness_functional_probability(ml, binary_episode):
    return ml.predict_proba(binary_episode)[0][0]


def state_abstraction(model, state1, state2, d):
    """
    This function compares to state, if they were in the same abstract class
    function returs 'True' otherwise 'False'
    """
    q_value1 = model.step_model.step([state1])
    q_value2 = model.step_model.step([state2])
    for i in range(len(q_value1[1][0])):
        print(q_value1[1][0][i])
        print(q_value2[1][0][i])
        if ceil(q_value1[1][0][i] / d) == ceil(q_value2[1][0][i] / d):
            continue
        else:
            return False
    return True


# def abstract_state(model, state1, d):
#     if type(state1) == str:
#         if state1 == 'done':
#             return 'end'
#     q_value1 = model.step_model.step([state1])
#     return (ceil(q_value1[1][0][0] / d), ceil(q_value1[1][0][1] / d))


# report function to check the performance metrics of the model
def report(model2, x_train, y_train, x_test, y_test):
    plt.ion()
    print("********************** reporting the result of the model **************************")
    print('The score for train data is {0}'.format(model2.score(x_train, y_train)))
    print('The score for test data is {0}'.format(model2.score(x_test, y_test)))

    predictions_train = model2.predict(x_train)
    predictions_test = model2.predict(x_test)

    print("\n\n--------------------------------------recall---------------------------------")

    print(
        'the test recall for the class yes is {0}'.format(metrics.recall_score(y_test, predictions_test, pos_label=1)))
    print('the test recall for the class no is {0}'.format(metrics.recall_score(y_test, predictions_test, pos_label=0)))

    print('the training recall for the class yes is {0}'.format(
        metrics.recall_score(y_train, predictions_train, pos_label=1)))
    print('the training recall for the class no is {0}'.format(
        metrics.recall_score(y_train, predictions_train, pos_label=0)))

    print("\n\n--------------------------------------precision------------------------------")

    print('the test precision for the class yes is {0}'.format(
        metrics.precision_score(y_test, predictions_test, pos_label=1)))
    print('the test precision for the class no is {0}'.format(
        metrics.precision_score(y_test, predictions_test, pos_label=0)))

    print('the training precision for the class yes is {0}'.format(
        metrics.precision_score(y_train, predictions_train, pos_label=1)))
    print('the training precision for the class no is {0}'.format(
        metrics.precision_score(y_train, predictions_train, pos_label=0)))

    print("\n\n")
    print(classification_report(y_test, predictions_test, target_names=['NO ', 'yes']))

    tn, fp, fn, tp = confusion_matrix(y_test, predictions_test).ravel()
    specificity = tn / (tn + fp)
    print("\n\nspecifity :", specificity)
    print("\n\n--------------------------------------confusion----------------------------")
    CM = metrics.confusion_matrix(y_test, predictions_test)
    print("The confusion Matrix:")
    print(CM)
    print('the accuracy score in {0}\n\n'.format(accuracy_score(y_test, predictions_test)))
    print("********************** plotting the confusion matrix & ROC curve **************************")
    ConfusionMatrixDisplay(CM, display_labels=model2.classes_).plot()
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions_test)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    display.plot()
    plt.pause(3)
    plt.ioff()


def fix_testing(testing_episodes, testing_states, Env2):
    buffer = []
    episodes_set = []
    j = 0
    for i in range(len(testing_episodes)):
        if type(testing_episodes[i][0]) is str and testing_episodes[i][0] == 'done':
            if i == 0:
                continue
            buffer.append(testing_episodes[i])
            episodes_set.append(buffer)
            buffer = []
        else:
            buffer.append(testing_episodes[i])
    if not (episodes_set[0][0][0] == Env2.set_state(testing_states[0])).all():
        del testing_states[0]
    if not (episodes_set[0][0][0] == Env2.set_state(testing_states[0])).all():
        assert False, 'problem in starting states'
    if len(episodes_set) != len(testing_states):
        del testing_states[-1]
    if len(episodes_set) != len(testing_states):
        assert False, 'problem in data prepration'
    return episodes_set, testing_states


def Abstract_classes(ep, abstraction_d, model):
    d = abstraction_d
    abs_states1 = []
    for episode in ep:
        for state, action in episode:
            if type(state) is str:
                continue
            abs_states1.append(state)
    abs_states1 = model.abstract_state(abs_states1, d)
    unique1 = list(set(abs_states1))
    uni1 = np.array(unique1)
    a = len(abs_states1)
    b = len(set(abs_states1))
    print("abstract states:", b)
    print("Concrete states", a)
    print("ratio", b / a)
    return unique1, uni1


def ML_first_representation(Abs_d, epsilon_functional_fault_boarder, uni1, model, ep, unique1):
    d = Abs_d
    epsilon = epsilon_functional_fault_boarder
    data1_x_b = []
    data1_y_b = []
    data1_y_f_b = []
    reward_fault_threshold = 200
    for episode in ep:
        record = np.zeros(len(uni1))

        if episode[-1][1] >= reward_fault_threshold:
            data1_y_b.append(0)
        else:
            data1_y_b.append(1)

        state_list = [k[0] for k in episode[:-1]]
        if is_fail_state(state_list):
            data1_y_f_b.append(1)
        else:
            data1_y_f_b.append(0)

        ab = model.abstract_state(state_list, d)
        for i in ab:
            record[unique1.index(i)] = 1
        data1_x_b.append(record)

    return data1_x_b, data1_y_b, data1_y_f_b


def translator(episode, model, d, unique5):
    """
    thid function takes the concrete episodes and returns the encoded episodes
    based on the presence and absence of the individuals
    :param 'episode': input episode
    :param 'model': RL model
    :param 'd': abstraction level = 1
    :param 'unique5': abstract classes
    :return: encoded episodse based on the presence and absence

    """
    d = d
    record = np.zeros(len(unique5))
    for state, action in episode:
        ab = model.abstract_state(state, d)
        if ab == 'end':
            continue
        if ab in unique5:
            ind = unique5.index(ab)
        record[ind] = 1
    return [record]


def transform(state):
    position = state[0]
    noise = np.random.uniform(low=0.95, high=1.05)
    new_position = position * noise
    new_state = deepcopy(state)
    new_state[0] = new_position
    # if new_position>2.4:
    # newstate = 2.4
    # if new_position<-2.4:
    # newstate = -2.4
    return new_state


def mutation_improved(population, model, env, objective_uncovered):
    """
    This is the final mutation function
    It takes the population as input and returns the mutated individual
    :param 'population': Population that we want to mutate
    :param 'model': RL model
    :param 'env': RL environment
    :param 'objective_uncovered: uncovered ubjectives for tournament selection
    :return: mutated candidate (we re-rexecute the episode from the mutation part)
    To-do:
    move deepcopy to the cadidate class methods .set info
    """
    parent = tournament_selection(population, 10, objective_uncovered)  # tournament selection
    parent1 = deepcopy(parent.get_candidate_values())
    if len(parent1) < 3:
        assert False, "parent in mutation is shorter than 3"
    Mutpoint = random.randint(3, (len(parent1) - 3))
    new_state = transform(parent1[Mutpoint][0])
    action = model.predict(new_state)
    if action != int(parent1[Mutpoint][1]):
        print('Mutation lured the agent ... ')
    new_parent = parent1[:Mutpoint]
    new_parent.append([new_state, 'Mut'])
    new_cand = Candidate(new_parent)
    new_cand.set_start_state(parent.get_start_state())

    re_executed_epis = re_execute(model, env, new_cand)

    re_executed_cand = Candidate(re_executed_epis)
    re_executed_cand.set_start_state(new_cand.get_start_state())
    re_executed_cand.set_info(deepcopy(parent.get_info()))
    re_executed_cand.set_info(["mutation is done! ", "mutpoint was:", Mutpoint])

    return re_executed_cand


def mutation_improved_p(parent, model, env, m_rate):
    """
    This is the final mutation function with input of a parent considering internal m_rate
    Here we give the parent to themutation funcion based on the given mutation
    rate of m_rate, we may mutate the episodes.
    :param 'parent' : individual that we want to mutate
    :param 'model': RL model
    :param 'env': RL environment
    :param 'm_rate': mutation : recommended value is 1/len(parent)
    :return : mutated individual
    To-do:
    move deepcopy to the cadidate .set info
    """
    # parent = tournament_selection(population, 10, objective_uncovered)  # tournament selection
    global MUTATION_NUMBER
    chance = random.uniform(0, 1)
    if chance > m_rate:
        return parent
    else:
        parent1 = deepcopy(parent.get_candidate_values())
        if len(parent1) < 3:
            assert False, "parent in mutation is shorter than 3"
        Mutpoint = random.randint(3, (len(parent1) - 3))
        new_state = transform(parent1[Mutpoint][0])
        action = model.predict(new_state)
        if action != int(parent1[Mutpoint][1]):
            print('Mutation lured the agent ... ')
        new_parent = parent1[:Mutpoint]
        new_parent.append([new_state, 'Mut'])
        new_cand = Candidate(new_parent)
        new_cand.set_start_state(parent.get_start_state())
        re_executed_epis = re_execute(model, env, new_cand)
        re_executed_cand = Candidate(re_executed_epis)
        re_executed_cand.set_start_state(new_cand.get_start_state())
        re_executed_cand.set_info(deepcopy(parent.get_info()))
        re_executed_cand.set_info(["mutation is done! ", "mutpoint was:", Mutpoint])
        MUTATION_NUMBER += 1
        return re_executed_cand


def Crossover_improved_v2(population, model, d, objective_uncovered):
    """
    This is the crossover function that we are using
    It takes the population as input and returns the mutated individual
    :param 'population': Population. we select a parent based on the tournament
     selection and then select the mutation point and then search for the matching point.
    :param 'model': RL model
    :param 'env': RL environment
    :param 'objective_uncovered: uncovered ubjectives for tournament selection
    :return: mutated candidate (we re-rexecute the episode from the mutation part)
    To-do:
    finding matching episode could be improved bu storing a mapping between concrete states and
    """
    found_match = False
    while not (found_match):
        parent = tournament_selection(population, 10, objective_uncovered)  # tournament selection
        parent1 = deepcopy(parent.get_candidate_values())
        parent1_start_point = deepcopy(parent.get_start_state())
        if len(parent1) < 4:
            assert False, 'input of crossover is shorter than expected '
        matches_list = []
        crosspoint = random.randint(1, (len(parent1) - 3))
        abs_class = list(model.abstract_state(parent1[crosspoint][0], d))
        for i in range(50):
            indx = random.randint(0, len(population) - 1)
            random_candidate = deepcopy(population[indx])
            random_cand_data = random_candidate.get_candidate_values()
            random_cand_start_point = random_candidate.get_start_state()
            for st_index in range(1, len(random_cand_data) - 3):
                random_ab = list(model.abstract_state(random_cand_data[st_index][0], d))
                if random_ab == abs_class:
                    matches_list.append(st_index)
                    found_match = True
            if found_match:
                break
                # print('Crossover. attemp',i)
    index_match_in_matchlist = random.randint(0, len(matches_list) - 1)
    matchpoint = matches_list[index_match_in_matchlist]
    match_candidate = deepcopy(random_candidate)
    match = deepcopy(random_cand_data)
    match_start = deepcopy(random_cand_start_point)
    offspring1 = deepcopy(parent1[:crosspoint])
    offspring1 += deepcopy(match[matchpoint:])
    offspring1[-1] = ['done', (len(offspring1) - 1)]
    candid1 = Candidate(offspring1)
    candid1.set_start_state(parent1_start_point)
    candid1.set_info(deepcopy(parent.get_info()))
    candid1.set_info(["crossover is Done!", "the crossover point is:", crosspoint])
    offspring2 = deepcopy(match[:matchpoint])
    offspring2 += deepcopy(parent1[crosspoint:])
    offspring2[-1] = ['done', (len(offspring2) - 1)]
    candid2 = Candidate(offspring2)
    candid2.set_start_state(match_start)
    candid2.set_info(deepcopy(match_candidate.get_info()))
    candid2.set_info(["crossover is Done!", "the crossover point is:", matchpoint])

    if len(offspring1) < 4:
        print(offspring1)
        assert False, 'created offspring 1 in crossover is shorter than expected '

    if len(offspring2) < 4:
        print(offspring2)
        assert False, 'created offspring 2 in crossover is shorter than expected '

    return candid1, candid2


def Crossover_improved_v2_random(population, model, d, objective_uncovered):
    found_match = False
    while not found_match:
        i = random.randint(0, len(population))
        parent1 = deepcopy(population[i].get_candidate_values())
        parent1_start_point = deepcopy(population[i].get_start_state())
        matches_list = []
        crosspoint = random.randint(1, (len(parent1) - 3))
        abs_class = list(model.abstract_state(parent1[crosspoint][0], d))
        attemp = 0
        for i in range(700):
            attemp += 1
            indx = random.randint(0, len(population) - 1)
            random_candidate = deepcopy(population[indx])
            random_cand_data = random_candidate.get_candidate_values()
            random_cand_start_point = random_candidate.get_start_state()
            for st_index in range(1, len(random_cand_data) - 3):
                random_ab = list(model.abstract_state(random_cand_data[st_index][0], d))
                if random_ab == abs_class:
                    matches_list.append(st_index)
                    found_match = True
            if found_match:
                break
    print("match found in --- attemps", attemp)
    index_match_in_matchlist = random.randint(0, len(matches_list) - 1)
    matchpoint = matches_list[index_match_in_matchlist]
    match_candidate = random_candidate
    match = random_cand_data
    match_start = deepcopy(random_cand_start_point)
    offspring1 = deepcopy(parent1[:crosspoint])
    offspring1 += deepcopy(match[matchpoint:])
    offspring1[-1] = ['done', (len(offspring1) - 1)]
    candid1 = Candidate(offspring1)
    candid1.set_start_state(parent1_start_point)

    offspring2 = deepcopy(match[:matchpoint])
    offspring2 += deepcopy(parent1[crosspoint:])
    offspring2[-1] = ['done', (len(offspring2) - 1)]
    candid2 = Candidate(offspring2)
    candid2.set_start_state(match_start)
    return candid1, candid2


def re_execute(model, env, candidate):
    obs = env.reset()
    obs = env.set_state(deepcopy(candidate.get_start_state()))
    episode = candidate.get_candidate_values()
    steps_to_mut_point = len(episode)
    episode_reward = 0.0
    done = False
    for i in range(steps_to_mut_point):
        action= model.predict(obs, deterministic=True)
        action_selected = episode[i][1]
        if action_selected == 'Mut':
            action_selected = model.predict(episode[i][0], deterministic=True)
            # break
        obs, reward, done, info = env.step(
            int(action_selected))  # its very important to select the action here it means that we may
        # follow the previous path until the mutation point or we follow the route that the trained agent wants to follow forcing vs following
        # env.render()
        episode_reward += reward
        if done:
            break
            # for j in range(200 - steps_to_mut_point): ###changed
    for j in range(500):
        if done:
            break
        action= model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        # env.reset = state1
        episode_reward += reward
        if reward > 201:
            assert False
    assert done
    if episode_reward > 501:
        assert False
    return info['mem'][-(int(episode_reward) + 1):]


def re_execution_improved(model, env, candidate):
    differences = []
    episode_limit = 500
    env.reset()
    obs = env.set_state(candidate.get_start_state())
    episode = candidate.get_candidate_values()
    # steps_to_mut_point = len(episode)
    episode_reward = 0.0
    for i in range(episode_limit):
        action= model.predict(obs, deterministic=True)
        action_selected = episode[i][1]
        if episode[i][0] == 'done':
            continue
        if i >= len(episode):
            action= model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            continue
        if action != int(action_selected):
            prob = model.action_probability(episode[i][0])
            differences.append([i, prob])
        obs, reward, done, info = env.step(int(action_selected))
        # env.render()
        # env.reset = state1
        episode_reward += reward
        if done:
            # assert not done
            break
    assert done, "not finished in 2oo steps "
    return differences


def re_execution_improved_v2(model, env, candidate):
    differences = []
    episode_limit = 500
    env.reset()
    obs = env.set_state(candidate.get_start_state())
    episode = candidate.get_candidate_values()
    episode_reward = 0.0
    for i in range(episode_limit):
        if i >= (len(episode) - 1):
            action= model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            if done:
                # assert not done
                print("Reward:", episode_reward)
                # break
                return differences
            continue
        action= model.predict(obs, deterministic=True)
        if episode[i][0] == 'done':
            print("first scenario, episode finished correctly")
            # continue
        print(len(episode), i)
        action_selected = episode[i][1]
        if action != int(action_selected):
            prob = model.action_probability(episode[i][0])
            differences.append([i, prob])
        obs, reward, done, info = env.step(int(action_selected))
        # env.render()
        # env.reset = state1
        episode_reward += reward
        if done:
            # assert not done
            break
    assert done, "not finished in 2oo steps "
    return differences


class Candidate:
    def __init__(self, candidates_vals):
        if isinstance(candidates_vals, (np.ndarray, np.generic)):
            self.candidate_values = candidates_vals.tolist()
        else:
            self.candidate_values = candidates_vals
        self.objective_values = []
        self.objectives_covered = []
        self.crowding_distance = 0
        self.uncertainity = []
        self.start_state = 0
        self.information = []
        self.mutation = False

    def get_candidate_values(self):
        return self.candidate_values

    def get_uncertainity_value(self, indx):
        return self.uncertainity[indx]

    def get_uncertainity_values(self):
        return self.uncertainity

    def set_uncertainity_values(self, uncertain):
        self.uncertainity = uncertain

    def set_candidate_values(self, cand):
        self.candidate_values = cand

    def set_candidate_values_at_index(self, indx, val):
        self.candidate_values[indx] = val

    def get_objective_values(self):
        return self.objective_values

    def get_objective_value(self, indx):
        return self.objective_values[indx]

    def set_objective_values(self, obj_vals):
        self.objective_values = obj_vals

    def add_objectives_covered(self, obj_covered):
        if obj_covered not in self.objectives_covered:
            self.objectives_covered.append(obj_covered)

    def get_covered_objectives(self):
        return self.objectives_covered

    def set_crowding_distance(self, cd):
        self.crowding_distance = cd

    def get_crowding_distance(self):
        return self.crowding_distance

    def exists_in_satisfied(self, indx):
        for ind in self.objectives_covered:
            if ind == indx:
                return True
        return False

    def is_objective_covered(self, obj_to_check):
        for obj in self.objectives_covered:
            if obj == obj_to_check:
                return True
        return False

    def set_start_state(self, start_point):
        self.start_state = deepcopy(start_point)

    def get_start_state(self):
        return self.start_state

    def set_info(self, new_information):
        self.information.append(new_information)

    def get_info(self):
        return self.information

    def mutated(self):
        self.mutation = True


def mutation_number_update(file_address, Mut_Num_to_add, iteration):
    if iteration == 0:
        with open(file_address, 'wb') as file:
            pickle.dump(Mut_Num_to_add, file)
        return
    with open(file_address, 'rb') as file2:
        Mut_num = pickle.load(file2)
    print(Mut_num)
    if type(Mut_num) == list:
        print('list')
        buffer = Mut_num
        buffer.append(Mut_Num_to_add)
        print(buffer)
    else:
        print('int')
        buffer = []
        buffer.append(Mut_num)
        buffer.append(Mut_Num_to_add)
        print(buffer)
    with open(file_address, 'wb') as file:
        pickle.dump(buffer, file)


# ##MOSA

scaler = preprocessing.StandardScaler()


# domination relation method, same as MOSA
def dominates(value_from_pop, value_from_archive, objective_uncovered):
    dominates_f1 = False
    dominates_f2 = False
    for each_objective in objective_uncovered:
        f1 = value_from_pop[each_objective]
        f2 = value_from_archive[each_objective]
        if f1 < f2:
            dominates_f1 = True
        if f2 < f1:
            dominates_f2 = True
        if dominates_f1 and dominates_f2:
            break
    if dominates_f1 == dominates_f2:
        return False
    elif dominates_f1:
        return True
    return False


# calculating the fitness value function

def evaulate_population(func, pop, parameters):
    for candidate in pop:
        if isinstance(candidate, Candidate):
            result = func(candidate.get_candidate_values())
            candidate.set_objective_values(result)
            # print(candidate.get_objective_values())


def evaulate_population_with_archive(func, pop, already_executed):
    to_ret = []
    for candidate in pop:
        if isinstance(candidate, Candidate):
            if candidate.get_candidate_values() in already_executed:
                continue

            result = func(candidate.get_candidate_values())
            candidate.set_objective_values(result)
            already_executed.append(candidate.get_candidate_values())
            to_ret.append(candidate)
    return to_ret


def exists_in_archive(archive, index):
    for candidate in archive:
        if candidate.exists_in_satisfied(index):
            return True
    return False


# searching archive
def get_from_archive(obj_index, archive):
    for candIndx in range(len(archive)):
        candidate = archive[candIndx]
        if candidate.exists_in_satisfied(obj_index):
            return candidate, candIndx
    return None


# updating archive with adding the number of objective it satisfies, Same as Mosa paper
def update_archive(pop, objective_uncovered, archive, no_of_Objectives, threshold_criteria):
    for objective_index in range(no_of_Objectives):
        for pop_index in range(len(pop)):
            objective_values = pop[pop_index].get_objective_values()
            # if not objective_values[objective_index] or not threshold_criteria[objective_index]:
            if objective_values[objective_index] <= threshold_criteria[objective_index]:
                if exists_in_archive(archive, objective_index):
                    archive_value, cand_indx = get_from_archive(objective_index, archive)
                    obj_archive_values = archive_value.get_objective_values()
                    if obj_archive_values[objective_index] > objective_values[objective_index]:
                        value_to_add = pop[pop_index]
                        value_to_add.add_objectives_covered(objective_index)
                        # archive.append(value_to_add)
                        archive[cand_indx] = value_to_add
                        if objective_index in objective_uncovered:
                            objective_uncovered.remove(objective_index)
                        # archive.remove(archive_value)
                else:
                    value_to_add = pop[pop_index]
                    value_to_add.add_objectives_covered(objective_index)
                    archive.append(value_to_add)
                    if objective_index in objective_uncovered:
                        objective_uncovered.remove(objective_index)


# method to get the most dominating one
def select_best(tournament_candidates, objective_uncovered):
    best = tournament_candidates[0]  # in case none is dominating other
    for i in range(len(tournament_candidates)):
        candidate1 = tournament_candidates[i]
        for j in range(len(tournament_candidates)):
            candidate2 = tournament_candidates[j]
            if (dominates(candidate1.get_objective_values(), candidate2.get_objective_values(), objective_uncovered)):
                best = candidate1
    return best


def tournament_selection_improved(pop, size, objective_uncovered):
    tournament_candidates = []
    for i in range(size):
        indx = random.randint(0, len(pop) - 1)
        random_candidate = pop[indx]
        tournament_candidates.append(random_candidate)

    best = select_best(tournament_candidates, objective_uncovered)
    return best;


def tournament_selection(pop, size, objective_uncovered):
    tournament_candidates = []
    for i in range(size):
        indx = random.randint(0, len(pop) - 1)
        random_candidate = pop[indx]
        tournament_candidates.append(random_candidate)

    best = select_best(tournament_candidates, objective_uncovered)
    return best


def generate_offspring_improved(population, model, env, d, objective_uncovered):
    population_to_return = []
    probability_C = 0.75
    probability_M = 0.3
    size = len(population)
    while (len(population_to_return) < size):
        probability_crossover = random.uniform(0, 1)
        if probability_crossover <= probability_C:  # 75% probability
            off1, off2 = Crossover_improved_v2(population, model, 1, objective_uncovered)
            population_to_return.append(off1)
            population_to_return.append(off2)
        probability_mutation = random.uniform(0, 1)
        if probability_mutation <= probability_M:  # 30% probability this in for test purposes
            off3 = mutation_improved(population, model, env, objective_uncovered)
            population_to_return.append(off3)
    return population_to_return


def generate_offspring_improved_v2(population, model, env, d, objective_uncovered):
    population_to_return = []
    probability_C = 0.75
    probability_M = 0.01
    size = len(population)
    while (len(population_to_return) < size):
        probability_crossover = random.uniform(0, 1)
        if probability_crossover <= probability_C:  # 75% probability
            parent1, parent2 = Crossover_improved_v2(population, model, d, objective_uncovered)
            parent1 = mutation_improved_p(parent1, model, env, (1 / len(parent1.get_candidate_values())))
            parent2 = mutation_improved_p(parent2, model, env, (1 / len(parent2.get_candidate_values())))
            population_to_return.append(parent1)
            population_to_return.append(parent2)

        if probability_crossover > probability_C:
            parent = tournament_selection(population, 10,
                                          objective_uncovered)  # we may add a very small number of duplicated individulas but its not important as we are removing them in the final executions
            population_to_return.append(
                mutation_improved_p(parent, model, env, (1 / len(parent.get_candidate_values()))))

    return population_to_return


def save_all_data(pop, no_of_Objectives, threshold_criteria, stored_data):
    '''
    This function will save all individulas with objective lower than treshhold

    '''
    threshold_criteria_to_add_to_archive = [200, 0.06, 0.05]
    # be careful here you can set the satisfiing objectives that based on them you want to store the data
    for individual in pop:
        individual_objective = individual.get_objective_values()
        for i in range(no_of_Objectives):
            if individual_objective[i] < threshold_criteria_to_add_to_archive[i]:
                # if individual not in stored_data:
                #   ind_ = deepcopy(individual)
                #   stored_data.append(ind_)
                # individual_objective_values = individual.get_objective_values()
                found = False
                for j in range(len(stored_data)):
                    if individual_objective == stored_data[j].get_objective_values():
                        found = True
                        break
                if not found:
                    ind_ = deepcopy(individual)
                    stored_data.append(ind_)
    # return stored_data


def save_all_data2(pop, stored_data):
    '''
    This function will save all individulas in generations
    you need to remove redundant data (based on fitness and ...)

    '''
    stored_data.append(list(pop))


def Build_Archive(pop, no_of_Objectives, threshold_criteria, stored_data, initial_population):
    '''
    If you are using the Archive of all generated episodes, this function
    removes the duplicated results and builds the Archive.
    :param 'pop': current generation
    :param 'no_of_Objectives': number of objectives
    :param 'threshold_criteria': threshold criteria (we are intrested in episodes that have fitness below these threshold values)
    :param 'stored_data': Archive of final episodes (return)
    :param 'initial_population': initial population. we are not considering these episodes in our archive for the second senario you need to add the number of faults, (implementation in RQ3)
    '''
    threshold_criteria_to_add_to_archive = threshold_criteria
    # be careful as we can have different values for criterias here to add episodes to archive and for GA stopping criteria
    for individual in pop:
        individual_objective = individual.get_objective_values()
        for i in range(no_of_Objectives):
            if individual_objective[i] < threshold_criteria_to_add_to_archive[i]:
                found = False
                for j in range(len(stored_data)):
                    if individual_objective == stored_data[j].get_objective_values():
                        found = True
                        break
                for k in range(len(initial_population)):
                    if individual_objective == initial_population[k].get_objective_values():
                        found = True
                        break
                if not found:
                    ind_ = deepcopy(individual)
                    stored_data.append(ind_)


# ###Sorting and RUN search
# finding best candidates and assigning to each front
def fast_dominating_sort(R_T, objective_uncovered):
    to_return = []
    front = []
    count = 0
    while len(R_T) > 1:
        count = 0
        for outer_loop in range(len(R_T)):
            best = R_T[outer_loop]
            add = True
            for inner_loop in range(len(R_T)):
                against = R_T[inner_loop]
                if best == against:
                    continue
                if (dominates(best.get_objective_values(), against.get_objective_values(), objective_uncovered)):
                    continue
                else:
                    add = False
                    break

            if add == True:
                if best not in front:
                    front.append(best)

                count = count + 1

        if len(front) > 0:
            to_return.append(front)
            for i in range(len(front)):
                R_T.remove(front[i])
                front = []

        if (len(to_return) == 0) or (count == 0):  # to check if no one dominates no one
            to_return.append(R_T)
            break

    return to_return


# sorting based on crowding distance
def sort_based_on_crowding_distance(e):
    values = e.get_crowding_distance()
    return values


def sort_based_on(e):
    values = e.get_objective_values()
    return values[0]


# sorting based on first objective value
def sort_worse(pop):
    pop.sort(key=sort_based_on, reverse=True)
    return pop


# preference sort, same as algorithm
def preference_sort(R_T, size, objective_uncovered):
    to_return = []
    for objective_index in objective_uncovered:
        min = 100
        best = R_T[0]
        for index in range(len(R_T)):
            objective_values = R_T[index].get_objective_values()
            if objective_values[objective_index] < min:
                min = objective_values[objective_index]
                best = R_T[index]
        to_return.append(best)
        R_T.remove(best)
    if len(R_T) > 0:
        E = fast_dominating_sort(R_T, objective_uncovered)
        for i in range(len(E)):
            to_return.append(E[i])
    return to_return


# converting to numpy array (Required by library)
def get_array_for_crowding_distance(sorted_front):
    list = []
    for value in sorted_front:
        objective_values = value.get_objective_values()

        np_array = numpy.array(objective_values)
        list.append(np_array)

    np_list = np.array(list)
    cd = calc_crowding_distance(np_list)
    return cd


# method to assign each candidate its crownding distance

def assign_crowding_distance_to_each_value(sorted_front, crowding_distance):
    for candidate_index in range(len(sorted_front)):
        objective_values = sorted_front[candidate_index]
        objective_values.set_crowding_distance(crowding_distance[candidate_index])


def run_search(func, initial_population, no_of_Objectives, criteria, archive, logger, start, time_budget, size, d, env,
               parameters, second_archive, gens):
    global MUTATION_NUMBER
    MUTATION_NUMBER = 0
    threshold_criteria = criteria
    objective_uncovered = []
    print("initial population ", type(initial_population), len(initial_population))

    for obj in range(no_of_Objectives):
        objective_uncovered.append(obj)  # initializing number of uncovered objective

    random_population = initial_population

    P_T = copy.copy(random_population)
    evaulate_population(func, random_population,
                        parameters)  # evaluating whole generation and storing results propabibly its with candidates

    print(random_population[0].get_objective_values())
    update_archive(random_population, objective_uncovered, archive, no_of_Objectives,
                   threshold_criteria)  # updating archive
    # save initial population
    save_all_data2(random_population, gens)
    iteration = 0
    # limit of number of generations
    while iteration < 10:
        iteration = iteration + 1  # iteration count
        # To-DO: limit by the time budget instead of the generation number
        for arc in archive:
            logger.info("***ARCHIVE***")
            logger.info("\nValues: " + str(
                arc.get_candidate_values()) + "\nwith objective values: " + str(
                arc.get_objective_values()) + "\nSatisfying Objective: " + str(
                arc.get_covered_objectives()))
        print("Iteration count: " + str(iteration))
        logger.info("Iteration is : " + str(iteration))
        logger.info("Number of mutations : " + str(MUTATION_NUMBER))

        R_T = []

        Q_T = generate_offspring_improved_v2(P_T, model, env, d,
                                             objective_uncovered)  # generate offsprings using crossover and mutation

        evaulate_population(func, Q_T, parameters)  # evaluating offspring
        update_archive(Q_T, objective_uncovered, archive, no_of_Objectives, threshold_criteria)  # updating archive
        save_all_data(Q_T, no_of_Objectives, threshold_criteria, second_archive)
        # save generations
        save_all_data2(Q_T, gens)
        R_T = copy.deepcopy(P_T)  # R_T = P_T union Q_T
        R_T.extend(Q_T)

        F = preference_sort(R_T, size, objective_uncovered)  # Preference sorting and getting fronts

        if len(objective_uncovered) == 0:  # checking if all objectives are covered
            print("all_objectives_covered")
            logger.info("***Final-ARCHIVE***")
            print(("***Final-ARCHIVE***"))
            for arc in archive:
                print("\nValues: " + str(
                    arc.get_candidate_values()) + "\nwith objective values: " + str(
                    arc.get_objective_values()) + "\nSatisfying Objective: " + str(
                    arc.get_covered_objectives()))

                logger.info("\nValues: " + str(
                    arc.get_candidate_values()) + "\nwith objective values: " + str(
                    arc.get_objective_values()) + "\nSatisfying Objective: " + str(
                    arc.get_covered_objectives()))
            logger.info("Iteration is : " + str(iteration))
            logger.info("Number of mutations : " + str(MUTATION_NUMBER))
            break

        P_T_1 = []  # creating next generatint PT+1
        index = 0

        while len(P_T_1) <= size:  # if length of current generation is less that size of front at top then add it

            if not isinstance(F[index], Candidate):
                if len(P_T_1) + len(F[index]) > size:
                    break
            else:
                if len(P_T_1) + 1 > size:
                    break

            front = F[index]
            if isinstance(F[index], Candidate):  # if front contains only one item
                P_T_1.append(F[index])
                F.remove(F[index])
            else:
                for ind in range(len(F[index])):  # if front have multiple items
                    val = F[index][ind]
                    P_T_1.append(val)

                F.remove(F[index])
        while (len(P_T_1)) < size:  # crowding distance
            copyFront = copy.deepcopy(F[index])
            sorted_front = sort_worse(copyFront)  # sort before crowding distance

            crowding_distance = get_array_for_crowding_distance(sorted_front)  # coverting to libaray compaitble array
            assign_crowding_distance_to_each_value(sorted_front,
                                                   crowding_distance)  # assinging each solution its crowding distance
            sorted_front.sort(key=sort_based_on_crowding_distance, reverse=True)  # sorting based on crowding distance

            if (len(sorted_front) + len(
                    P_T_1)) > size:  # maintaining length and adding solutions with most crowding distances
                for sorted_front_indx in range(len(sorted_front)):
                    candidate = sorted_front[sorted_front_indx]
                    P_T_1.append(candidate)
                    if len(P_T_1) >= size:
                        break

            index = index + 1

        P_T_1 = P_T_1[0:size]
        P_T = P_T_1  # assigning PT+1 to PT


def minimize(func, population, lb, ub, no_of_Objectives, criteria, time_budget, logger, archive, size, d, env,
             parameters, second_archive, gens):
    assert hasattr(func, '__call__')

    start = time.time()
    run_search(func, population, no_of_Objectives, criteria, archive, logger, start, time_budget, size, d, env,
               parameters, second_archive, gens)


class CartPole_caseStudy():
    def __init__(self):
        logger = logging.getLogger()

        now = datetime.now()
        log_file = 'output/STARLA' + str(i) + '_V2' + str(now) + '.log'
        logging.basicConfig(filename=log_file,
                            format='%(asctime)s %(message)s')
        self.parameters = [model, d, unique5]
        logger.setLevel(logging.DEBUG)

    def _evaluate(self, x):
        fv = x
        model, d, unique5 = self.parameters
        obj1 = fitness_reward(fv)
        obj2 = fitness_confidence(fv, model, 'm')
        binary_fv = translator(fv, model, d, unique5)
        obj3 = fitness_functional_probability(RF_FF_1rep, binary_fv)
        # obj4 = fitness_functional_probability(RF_RF_1rep,binary_fv)
        to_ret = [obj1, obj2, obj3]
        logger = logging.getLogger()
        logger.info(str(fv) + "," + str(to_ret))
        return to_ret


def run(i, population, archive, second_archive, gens):
    env = env2
    d = DD
    size = len(population)
    lb = [0, 0, 0]
    ub = [100000, 1000000, 100000]

    parameters = [model, d, unique1]
    threshold_criteria = [200, 0.04, 0.05]

    no_of_Objectives = 3
    print("1", type(population), len(population))

    now = datetime.now()
    global logger
    logger = logging.getLogger()
    log_file = os.path.dirname(__file__)
    log_file += '/log/STARLA' + str(i) + '_V1' + str(now) + '.log'
    logging.basicConfig(filename=log_file,
                        format='%(asctime)s %(message)s')

    logger.setLevel(logging.DEBUG)

    archive = minimize(CartPole_caseStudy()._evaluate, population, lb, ub,
                       no_of_Objectives, threshold_criteria, 7200,
                       logger, archive, size, d, env2, parameters, second_archive, gens)
    logger.info("Iteration completed")
    logger.info("mu" + str(MUTATION_NUMBER))


# ###analyzer
def analyze_result(result):
    '''
    this function is to aggrigate the differences of the results
    :param `result`: this is the output of the re-execution-improved function
    :return ``:
    '''
    total_dif = 0
    # store_diff=[]
    for i in range(len(result)):
        dif = abs(result[i][1][0] - result[i][1][1])
        # store_diff.append([i,dif])
        total_dif += dif
    return total_dif  # , store_diff


def get_objective_distribution_and_set_candidate_objectives(population, model, d,
                                                            unique1, RF_FF_1rep,
                                                            RF_RF_1rep):
    fit1_list = []
    fit2_list = []
    fit3_list = []
    fit4_list = []
    for i in range(len(population)):
        ind_data = population[i].get_candidate_values()
        fit1 = fitness_reward(ind_data)
        fit2 = fitness_confidence(ind_data, model, 'm')
        binary_fv = translator(ind_data, model, d, unique1)
        fit3 = fitness_functional_probability(RF_FF_1rep, binary_fv)
        fit4 = fitness_reward_probability(RF_RF_1rep, binary_fv)
        obj = [fit1, fit2, fit3, fit4]
        population[i].set_objective_values(obj)
        fit1_list.append(fit1)
        fit2_list.append(fit2)
        fit3_list.append(fit3)
        fit4_list.append(fit4)
    return fit1_list, fit2_list, fit3_list, fit4_list


def get_objective_distribution(population, model, d, unique1, RF_FF_1rep, RF_RF_1rep):
    fit1_list = []
    fit2_list = []
    fit3_list = []
    fit4_list = []
    for i in range(len(population)):
        ind_data = population[i].get_candidate_values()
        fit1 = fitness_reward(ind_data)
        fit2 = fitness_confidence(ind_data, model, 'm')
        binary_fv = translator(ind_data, model, d, unique1)
        fit3 = fitness_functional_probability(RF_FF_1rep, binary_fv)
        fit4 = fitness_reward_probability(RF_RF_1rep, binary_fv)
        # obj = [fit1,fit2,fit3,fit4]
        # population[i].set_objective_values(obj)
        fit1_list.append(fit1)
        fit2_list.append(fit2)
        fit3_list.append(fit3)
        fit4_list.append(fit4)
    return fit1_list, fit2_list, fit3_list, fit4_list


def was_in_initial_population(solution, population, no_of_Objectives):
    flag = False
    for individuals_ in population:
        if individuals_.get_objective_values() == solution.get_objective_values():
            flag = True
    if not flag:
        return solution
    if flag:
        return 0


def analyze_set_differences(differences_set):
    '''
    input is a set of differences
    '''
    analyzed_results = []
    for item in differences_set:
        res = [len(item[0]), analyze_result(item[0]), item[1], len(item[0]) / item[1]]
        analyzed_results.append(res)
    return analyzed_results


def extract_differences(solution_set):
    '''
    input is a set of solutions like archive or second_archive
    the output a list ([list of differences as a result of re-execution],reward)
    '''
    differences = []
    for dastan in solution_set:
        reward = dastan.get_objective_values()[0]
        differences.append([re_execution_improved_v2(model, env2, dastan), reward])
    return differences


def get_results_distribution(results):
    num_of_diff = []
    diff_confi = []
    diff_ration = []
    for item in results:
        num_of_diff.append(item[0])
        diff_confi.append(item[1])
        diff_ration.append(item[3])
    return num_of_diff, diff_confi, diff_ration


def random_test_1(model, env, Num):
    obs = env.reset()
    counter = 1
    episode_reward = 0.0
    for i in range(Num):
        action = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            counter += 1
            end = i
            print("Reward:", episode_reward, "final state", info['mem'][-2][0])
            episode_reward = 0.0
            obs = env.reset()
    iter = deepcopy(counter)
    u = 1
    while iter > 1:
        if info['mem'][-u][0] == 'done':
            lastpoint = -u
            iter -= 1
        u += 1
    fin = Num - end
    start = -Num - counter
    randomtest = info['mem'][lastpoint:-fin]
    ran_state = info['state'][(-counter + 1):-1]
    return randomtest, ran_state


def random_test_2(model, env, Num):
    obs = env.reset()
    counter = 1
    episode_reward = 0.0
    for i in range(Num):
        action = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # if is_fail_state(obs):
        #     print(obs[2] - obs[0])
        # if reward < 0:
        #     time.sleep(0.1)
        episode_reward += reward
        if done:
            counter += 1
            end = i
            episode_reward = 0.0
            obs = env.reset()
    iter = deepcopy(counter)
    u = 1
    while iter > 1:
        if type(info['mem'][-u][0]) is str and info['mem'][-u][0] == 'done':
            lastpoint = -u
            iter -= 1
        u += 1
    fin = Num - end
    start = -Num - counter
    randomtest = info['mem'][lastpoint:-fin]
    ran_state = info['state'][(-counter + 1):-1]
    return randomtest, ran_state


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


#########################################################  Read DATA and Load Model #############
# ##Model and Data
# Address of the trained RL model use the dataset provided with replication package
start_time = time.time()
Drive_model = '../result/catcher/model/catcher_model.pkl'
model = TorchModel(torch.load(Drive_model).cuda().eval())

env2 = PLEStoreWrapper(PLE(TestCatcher(width=64*8, height=64*8, init_lives=1), display_screen=DISPLAY_SCREEN))

d = DD

def ml_model(uni1, model, ep, unique1):
    d = DD
    epsilon = 5
    data1_x_b, data1_y_b, data1_y_f_b = ML_first_representation(d, epsilon, uni1, model, ep, unique1)
    print(f'\033[32mdata lenth {len(data1_x_b)} with {np.array(data1_y_b).sum()} rf and {np.array(data1_y_f_b).sum()} ff\033[0m')

    #########################################################  Train ML -  Reward fault predictor  #############

    X_train_reward_fault, X_test_reward_fault, y_train_reward_fault, y_test_reward_fault = train_test_split(data1_x_b,
                                                                                                            data1_y_b,
                                                                                                            test_size=0.2,
                                                                                                            random_state=42)

    RF_RF_1rep = RandomForestClassifier(random_state=0, class_weight='balanced')
    RF_RF_1rep.fit(X_train_reward_fault, y_train_reward_fault)
    # report(RF_RF_1rep, X_train_reward_fault, y_train_reward_fault, X_test_reward_fault, y_test_reward_fault)

    #########################################################  Train ML - Functional fault predictor #############


    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(data1_x_b, data1_y_f_b, test_size=0.2, random_state=42)
    RF_FF_1rep = RandomForestClassifier(random_state=0, class_weight='balanced')
    RF_FF_1rep.fit(X_train_f, y_train_f)
    report(RF_FF_1rep, X_train_f, y_train_f, X_test_f, y_test_f)
    return RF_RF_1rep, RF_FF_1rep

# #Run
"""
The following annotation codes are for testing ML models. 
Uncommenting them will only test ML models and no results will be generated.
"""

MUTATION_NUMBER = 0  # set the mutation counter to 0
Run_number = 2
for s in range(10):
    print(f'epoch s{s} begin')
# print('catcher')
# for d in [1, 0.5, 0.1, 0.05]:
    ee, qq = random_test_2(model, env2, 800_000)
    test, teststate = fix_testing(ee, qq, env2)
    unique1, uni1 = Abstract_classes(test, d, model)
    unique5 = unique1

    # data1_x_b, data1_y_b, data1_y_f_b = ML_first_representation(d, None, uni1, model, test, unique1)
    # X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(data1_x_b, data1_y_f_b, test_size=0.2, random_state=42)
    # RF_FF_1rep = RandomForestClassifier(random_state=0, class_weight='balanced')
    # RF_FF_1rep.fit(X_train_f, y_train_f)
    # predictions_test = RF_FF_1rep.predict(X_test_f)
    # TP, FN, FP, TN = 0, 0, 0, 0
    # for idx in range(len(y_test_f)):
    #     if y_test_f[idx] == 1:
    #         if predictions_test[idx] == 1:
    #             TP += 1
    #         else:
    #             FN += 1
    #     else:
    #         if predictions_test[idx] == 1:
    #             FP += 1
    #         else:
    #             TN += 1
    # P = TP/(TP+FP)
    # R = TP/(TP+FN)
    # print('Accuracy\tPrecision\tRecall\tF1')
    # print(f'{(TP+TN)/len(y_test_f)}\t{P}\t{R}\t{2*P*R/(P+R)}')
    # continue

    RF_RF_1rep, RF_FF_1rep = ml_model(uni1, model, test, unique1)
    print('len population', len(test))
    start_state_ep1 = teststate
    ep1 = test
    population = []
    for i in range(0, 1500):  # size of the initial population is 1500
        if len(ep1[i]) < 200:
            continue
        cd = Candidate(ep1[i])
        cd.set_start_state(start_state_ep1[i])
        population.append(cd)
    archive1 = []
    second_arch1 = []
    generations = []  # all of the episodes generated during the search
    run(0, population, archive1, second_arch1, generations)

    dirname = os.path.dirname(__file__)
    with open(
            dirname + f'/Results/May17_arch1_r110_rt200_population1500lastfull_run{Run_number}_{s}.pickle',
            'wb') as file:
        pickle.dump(archive1, file)
    with open(
            dirname + f'/Results/May17_second_arch1_r110_rt200_population1500lastfull_run{Run_number}_{s}.pickle',
            'wb') as file:
        pickle.dump(second_arch1, file)
    with open(
            dirname + f'/Results/May17_generations_r110_rt200_population1500lastfull_run{Run_number}_{s}.pickle',
            'wb') as file:
        pickle.dump(generations, file)
    mutation_number_update(dirname + f'/Results/Mutation_number_run{Run_number}.pickle', MUTATION_NUMBER,
                           s)
