import pickle
import os
import numpy as np
from copy import deepcopy


def change_threshold_in_similarity_data(new_threshold, results):
    return_set = []
    for re_exe in results:
        episode = re_exe[4]
        ff_prob = episode.get_objective_values()[2]
        if ff_prob <= new_threshold:
            return_set.append(re_exe)
    return return_set


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


items = os.listdir('.')
avg_rate = []
for re_exe in items[:]:
    # print("\n\n-----------------------------------------------------\n\n")
    if re_exe[:11] == "re_executed":
        final_consistent_ff_count = 0
        print(f'{re_exe}')
        with open(f'./{re_exe}', 'rb') as file2:
            data = pickle.load(file2)
        new_data = change_threshold_in_similarity_data(0.5, data)
        for result in new_data:
            inconsistent, div, ff, states, episode = result
            if ff:
                if not inconsistent:
                    final_consistent_ff_count += 1
        print(f'\033[31mfailure {final_consistent_ff_count} with {len(new_data)} in sum\033[0m')
        print(f'\033[31mfailure rate {final_consistent_ff_count/len(new_data)}\033[0m')
        avg_rate.append(final_consistent_ff_count/len(new_data))
print(f'average failure rate {np.average(avg_rate)}')
