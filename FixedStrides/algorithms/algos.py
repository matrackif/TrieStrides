from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np
from copy import deepcopy
from utils.utils import *


# From the Wiley document by Wolant and Kaczmarski
# TODO verify this way also works and create RC array
def get_node_counts2(prefixes: List[str]):
    prefixes = sorted(prefixes, key=binary_to_int)
    max_len = max(len(p) for p in prefixes)
    print_d('Prefixes: ' + str(prefixes))
    P = []
    L = []
    F = []
    C = []
    for i in range(max_len):
        P.append([])
        L.append([])
        F.append([])
        C.append([])
        for j in range(len(prefixes)):
            if len(prefixes[j]) > i:
                P[i].append(prefixes[j][i])
            else:
                P[i].append(None)
            F[i].append(1 if j == 0 else 0)
            L[i].append(len(prefixes[j]))
            C[i].append(1 if i < len(prefixes[j]) else 0)
    for i in range(1, max_len):
        for j in range(1, len(prefixes)):
            if P[i - 1][j - 1] is not None and P[i - 1][j] is not None:
                F[i][j] = int(P[i - 1][j - 1] != P[i - 1][j]) or F[i - 1][j]
    S = deepcopy(F)
    for i in range(len(S)):
        S[i] = np.cumsum(S[i]).tolist()

    print_d('P: ' + str(P) + '\nL: ' + str(L) + '\nF: ' + str(F) + '\nS: ' + str(S))


# def fixed_strides(prefixes: List[str]):
#     max_len = max(len(p) for p in prefixes)
#     k = max_len
#     c = {}
#     m = {}
#     nodes = get_node_counts(prefixes)
#
#     c[-1] = [0] * k
#     for i in range(max_len):
#         c[i] = [(2 ** (i + 1)) if j == 1 else 0 for j in range(max_len + 1)]
#         m[i] = [-1 if j == 1 else 0 for j in range(max_len + 1)]
#
#     print_d('nodes: ' + str(nodes) + ' c:' + str(c) + '\n' ' m:' + str(m))
#     for r in range(2, k + 1):
#         for j in range(r - 1, max_len):
#             min_j = max(m[j - 1][r], m[j][r - 1])
#             print_d('m: ' + str(m) + 'min_j: ' + str(min_j))
#             print_d('c: ' + str(c))
#             min_cost = c[j][r - 1]
#             min_l = m[j][r - 1]
#             for z in range(min_j, j):
#                 print_d('r: ' + str(r) + ' j: ' + str(j) + ' z: ' + str(z))
#                 cost = c[z][j - 1] + (nodes[z + 1] * (2 ** (j - z)))
#                 if cost < min_cost:
#                     min_cost = cost
#                     min_l = z
#             c[j][r] = min_cost
#             m[j][r] = min_l
#     print_d('------------------------')
#     print_d('c:' + str(c) + '\n' ' m:' + str(m))
#     ############################################
#
#     strides = []
#     levels_to_cover = max_len - 1  # cover levels 0 through "levels_to_cover" (levels_to_cover + 1 = max_len)
#     tmp = 0
#     while levels_to_cover >= 0:
#         min_m = m[levels_to_cover][levels_to_cover + 1]
#         tmp += c[levels_to_cover][levels_to_cover + 1]
#         stride = levels_to_cover - min_m
#         levels_to_cover -= stride
#         strides.append(stride)
#
#     strides = strides[-1::-1]  # reverse strides array
#     # print('c[max_len - 1][max_len]:', c[max_len - 1][max_len])
#     return strides, nodes


def fixed_strides(prefixes: List[str], k: int = None):
    max_len = max(len(p) for p in prefixes)
    c = {}
    m = {}
    nodes = get_node_counts(prefixes)

    if k is None:
        k = max_len

    c[-1] = [0] * k
    for i in range(max_len):
        c[i] = [(2 ** (i + 1)) if j == 1 else 0 for j in range(max_len + 1)]
        m[i] = [-1 if j == 1 else 0 for j in range(max_len + 1)]

    for j in range(1, max_len):
        for r in range(2, k + 1):
            min_j = max(m[j - 1][r], m[j][r - 1])
            min_cost = c[j][r - 1]
            min_l = m[j][r - 1]
            for z in range(min_j, j):
                cost = c[z][j - 1] + (nodes[z + 1] * (2 ** (j - z)))
                if cost < min_cost:
                    min_cost = cost
                    min_l = z
            c[j][r] = min_cost
            m[j][r] = min_l
    ############################################

    strides = []
    levels_to_cover = max_len - 1  # cover levels 0 through "levels_to_cover" (levels_to_cover + 1 = max_len)
    tmp_k = k
    while levels_to_cover >= 0:
        min_m = m[levels_to_cover][tmp_k]
        stride = levels_to_cover - min_m
        tmp_k -= stride
        levels_to_cover -= stride
        strides.append(stride)

    strides = strides[-1::-1]  # reverse strides array
    return strides, nodes


# Algorithms from Efficient Construction of Pipelined Multibit-Trie Router-Tables by Kun Suk Kim & Sartaj Sahni 2007
def greedy_cover(k, p, nodes):
    max_len = len(nodes)
    if k is None or k == 0:
        k = max_len

    stages = level = 0
    while stages < k:
        i = 1
        while (nodes[level] * (2 ** i)) <= p and (level + i <= max_len):
            i += 1
        if level + i > max_len:
            return True
        if i == 1:
            return False
        level += i - 1
        stages += 1

    return False


def binary_search_mms(k, p, nodes):
    if greedy_cover(k, p, nodes):
        return p
    p *= 2
    while not greedy_cover(k, p, nodes):
        p *= 2
    low = int(p / 2) + 1
    high = p
    p = int((low + high) / 2)
    while low < high:
        if greedy_cover(k, p, nodes):
            high = p
        else:
            low = p + 1
        p = int((low + high) / 2)
    return high


def get_max_mem_per_level(nodes: List[int], num_levels=0):
    # Initially p is nodes(l) ∗ 2 where level l has
    # the max number of nodes in the 1-bit trie.
    p = max(node_count for node_count in nodes) * 2
    max_mem = binary_search_mms(num_levels, p, nodes)
    print('binary_search_mms with k={} and p={} yielded max memory of: {}'.format(num_levels, p, max_mem))
    return max_mem


def init_fixed_strides_2_dicts(c: Dict[Tuple, int], m: Dict[Tuple, int], max_lvls: int):
    for i in range(max_lvls):
        for j in range(max_lvls):
            for r in range(1, max_lvls + 1):
                if (j - i + 1) >= r:
                    c[i, j, r] = 0
                    m[i, j, r] = 0


def fixed_strides_2_impl(c: Dict[Tuple, int], m: Dict[Tuple, int], s: int, f: int, num_levels: int, max_mem_per_lvl: int, nodes: List[int]):
    INF = np.iinfo(np.uint32).max
    for j in range(s, f + 1):
        c[s, j, 1] = nodes[s] * (2 ** (j - s + 1))
        if c[s, j, 1] > max_mem_per_lvl:
            c[s, j, 1] = INF
        m[s, j, 1] = s - 1

    for r in range(2, num_levels + 1):
        for j in range(s + r - 1, f + 1):
            min_cost = INF
            min_l = INF
            for z in range(j - 1, s + r - 3, -1):
                cost = None
                if nodes[z + 1] * (2 ** (j - z)) <= max_mem_per_lvl:
                    cost = c[s, z, r - 1] + nodes[z + 1] * (2 ** (j - z))
                else:
                    break
                if cost < min_cost:
                    min_cost = cost
                    min_l = z
            c[s, j, r] = min_cost
            m[s, j, r] = min_l


# Algorithms from Efficient Construction of Pipelined Multibit-Trie Router-Tables by Kun Suk Kim & Sartaj Sahni 2007
def fixed_strides_2(prefixes: List[str], num_levels: int = 0):

    nodes = get_node_counts(prefixes)
    find_min_cost_tree = False
    if num_levels is None or num_levels == 0:
        num_levels = len(nodes)
        find_min_cost_tree = True
    max_len = len(nodes)
    max_mem_per_lvl = get_max_mem_per_level(nodes, num_levels)
    c = dict()
    m = dict()
    init_fixed_strides_2_dicts(c, m, max_len)
    fixed_strides_2_impl(c, m, 0, max_len - 1, num_levels, max_mem_per_lvl, nodes)

    strides = []
    levels_to_cover = max_len - 1  # cover levels 0 through "levels_to_cover" (levels_to_cover + 1 = max_len)
    tmp_k = num_levels
    # If finding minimal pipelined tree, then find how many levels gives us the smallest cost of tree
    if find_min_cost_tree:
        min = c[0, max_len - 1, 1]
        min_lvl = 1
        for i in range(2, max_len):
            if c[0, max_len - 1, i] < min:
                min = c[0, max_len - 1, i]
                min_lvl = i
        tmp_k = min_lvl

    while levels_to_cover >= 0:
        min_m = m[0, levels_to_cover, tmp_k]
        # tmp += c[levels_to_cover][levels_to_cover + 1]
        stride = levels_to_cover - min_m
        tmp_k -= 1
        levels_to_cover -= stride
        strides.append(stride)

    strides = strides[-1::-1]  # reverse strides array
    return strides, nodes


# Algorithm that seeks to store equal number of keys at each level of the multi-bit tree
# Finds the strides giving such a tree
# Takes into account the distribution of prefix lengths
# TODO assert all prefixes are unique?
def distribute_prefixes(prefixes: List[str], num_levels: int):
    lengths = get_lengths(prefixes)
    lengths_cum_sum = [0] * len(lengths)
    for i in range(len(lengths)):
        if i == 0:
            lengths_cum_sum[i] = lengths[i]
        else:
            lengths_cum_sum[i] = lengths[i] + lengths_cum_sum[i - 1]
    max_len = len(lengths)
    num_prefixes = len(prefixes)
    if num_levels > max_len:
        raise Exception("Number of levels: {} is greater than max length of prefixes: {}".format(num_levels, max_len))
    num_prefixes_per_lvl = int(num_prefixes / num_levels)
    remainder = num_prefixes % num_levels
    prefixes_per_lvl = [num_prefixes_per_lvl + 1 if i < remainder else num_prefixes_per_lvl for i in range(num_levels)]
    intervals = []

    i, interval_begin = 0, 0
    prefixes_covered_idx = 0
    prefixes_covered = prefixes_per_lvl[prefixes_covered_idx]
    while i < max_len:
        while lengths_cum_sum[i] >= prefixes_covered:
            intervals.append((interval_begin, i))
            # Distinguish the case where we cover the interval exactly with no prefixes left over
            # Then we will have non-overlapping intervals
            interval_begin = i + 1 if lengths_cum_sum[i] == prefixes_covered else i
            if prefixes_covered >= num_prefixes:
                break
            prefixes_covered_idx += 1
            prefixes_covered += prefixes_per_lvl[prefixes_covered_idx]
        i += 1
    # print('distribute_prefixes() found intervals:', intervals)
    strides = []
    return strides


def equal_level_strides(prefixes: List[str], num_levels: int):
    lengths = get_lengths(prefixes)
    lengths_cpy = deepcopy(lengths)
    max_len = len(lengths)

    num_prefixes = len(prefixes)
    if num_levels > max_len:
        raise Exception("Number of levels: {} is greater than max length of prefixes: {}".format(num_levels, max_len))
    elif num_levels == 0:
        raise Exception("Numbers of levels cannot be 0")
    num_prefixes_per_lvl = int(num_prefixes / num_levels)
    remainder = num_prefixes % num_levels
    prefixes_per_lvl = [num_prefixes_per_lvl + 1 if i < remainder else num_prefixes_per_lvl for i in range(num_levels)]

    # Handle some edge cases
    if num_levels == max_len:
        return [1] * max_len
    elif num_levels == 1:
        return [max_len]

    intervals = []
    i, interval_begin = 0, 0
    prefixes_covered_idx = 0
    prefixes_to_cover = prefixes_per_lvl[prefixes_covered_idx]
    # prefixes_covered = 0
    prefixes_remaining = prefixes_to_cover
    while i < max_len:
        # handle case where all of remaining prefixes will have the same length
        num_prefixes_in_remaining_levels = 0
        for j in range(prefixes_covered_idx, len(prefixes_per_lvl)):
            num_prefixes_in_remaining_levels += prefixes_per_lvl[prefixes_covered_idx]
        if num_prefixes_in_remaining_levels == lengths_cpy[i] and prefixes_covered_idx < len(prefixes_per_lvl) - 1:
            lengths_cpy[0] += lengths_cpy[1]
            for k in range(1, i):
                if k == i - 1:
                    lengths_cpy[k] = 0
                    break
                lengths_cpy[k] = lengths_cpy[k + 1]

        if lengths_cpy[i] > prefixes_remaining:
            half_of_lengths = int(lengths_cpy[i] / 2)
            # TODO handle edge cases when we can't shift to right or left
            if prefixes_remaining <= half_of_lengths:
                if i == 0:
                    tmp_sum = lengths_cpy[0] - prefixes_remaining
                    sum = 0
                    num_lvls = 0
                    j = 1
                    while sum < tmp_sum:
                        sum += prefixes_per_lvl[j]
                        num_lvls += 1
                        j += 1
                    lengths_cpy[num_levels - 2] += tmp_sum
                    lengths_cpy[0] = prefixes_remaining
                else:
                    # Shift remaining to left:
                    lengths_cpy[i] -= prefixes_remaining
                    lengths_cpy[i - 1] += prefixes_remaining
            else:
                # Shift remaining to right:
                lengths_cpy[i] -= prefixes_remaining
                lengths_cpy[i + 1] += prefixes_remaining
            intervals = []
            i, interval_begin = 0, 0
            prefixes_covered_idx = 0
            prefixes_to_cover = prefixes_per_lvl[prefixes_covered_idx]
            prefixes_remaining = prefixes_to_cover
            continue
        prefixes_remaining -= lengths_cpy[i]
        if prefixes_remaining == 0:
            intervals.append((interval_begin, i))
            if len(intervals) == num_levels:
                break
            prefixes_covered_idx += 1
            prefixes_to_cover = prefixes_per_lvl[prefixes_covered_idx]
            prefixes_remaining = prefixes_to_cover
            interval_begin = i + 1
        i += 1
    strides = []
    for interval in intervals:
        strides.append(interval[1] - interval[0] + 1)
    return strides


class ConfigurationGenerator(random.Random):

    def __init__(self, seed_val=0, max_len=32, max_stride_val: int = 22, shuffle: bool = False):
        # Using the name seed_value, in order for it not to be confused with the seed() method in Random class
        super().__init__(seed_val)
        self.seed_value = seed_val
        self.max_len = max_len
        self.configs = {}
        self.max_stride_value = max_stride_val
        self.shuffle = shuffle

    def gen_config(self, num_levels: int):

        if num_levels == 1:
            return [self.max_len]
        elif num_levels == self.max_len:
            return [1] * self.max_len

        # Make sure we have a valid max stride value, one that will allow us to cover max_len bits given our tree height
        assert (self.max_stride_value >= int(self.max_len / num_levels + 0.5))

        config = []
        levels_remaining = num_levels
        end_range = self.max_len - levels_remaining + 1
        # The intervals for possible stride values get smaller and smaller
        # This often means that the end of the configuration is full of smaller numbers
        # In order to mitigate this effect, we randomize the interval we choose the random number from
        end_range = self.randint(1, end_range)
        end_range = end_range if end_range <= self.max_stride_value else self.max_stride_value
        bits_to_cover = self.max_len
        while levels_remaining > 0:
            stride = 0
            if levels_remaining > 1:
                stride = super().randint(1, end_range)
            elif levels_remaining == 1:
                stride = bits_to_cover
                if stride > self.max_stride_value:
                    # Distribute evenly among previous strides
                    remainder = stride - self.max_stride_value
                    stride = self.max_stride_value
                    idx = 0
                    while remainder > 0:
                        if idx > len(config) - 1:
                            idx = 0
                        config[idx] += 1
                        remainder -= 1
                        idx += 1

            config.append(stride)
            bits_to_cover -= stride
            levels_remaining -= 1
            end_range = bits_to_cover - levels_remaining + 1
            end_range = self.randint(1, end_range)
            end_range = end_range if end_range <= self.max_stride_value else self.max_stride_value
        if self.shuffle:
            super().shuffle(config)
        return config

    def gen_unique_config(self, num_levels: int, num_configs_per_level: int):
        if num_levels not in self.configs.keys():
            self.configs[num_levels] = []
        for _ in range(num_configs_per_level):
            attempts = 0
            config = self.gen_config(num_levels)
            while config in self.configs[num_levels] and attempts < num_configs_per_level:
                config = self.gen_config(num_levels)
                attempts += 1
            if attempts < num_configs_per_level:
                self.configs[num_levels].append(config)

    def gen_configs(self, min_num_levels: int, max_num_levels: int, num_configs_per_level: int):
        self.configs = {}
        for i in range(min_num_levels, max_num_levels + 1):
            self.gen_unique_config(i, num_configs_per_level)
        return self.configs


def gen_random_configs(filename: str = "ip32_random.json", min_num_levels: int = 3, max_num_levels: int = 10,
                           num_configs_per_level: int = 100, max_len: int = 32, seed: int = 0, max_stride_val: int = 20, shuffle: bool = False):
    result_file_name = "ip32_random_shuffled.csv" if shuffle else "ip32_random_results.csv"
    c = ConfigurationGenerator(seed_val=seed, max_len=max_len, max_stride_val=max_stride_val, shuffle=shuffle)
    configs = c.gen_configs(min_num_levels, max_num_levels, num_configs_per_level)
    configs_as_list_of_lists = []
    for num_strides in configs.keys():
        for config in configs[num_strides]:
            configs_as_list_of_lists.append(config)
    configs_to_json(filename, result_file_name, configs_as_list_of_lists)


if __name__ == '__main__':
    gen_random_configs(filename="ip32_random_shuffled.json", max_num_levels=32, max_stride_val=24, shuffle=True)


class BruteForceConfigGenerator:

    def __init__(self, max_len: int = 32, min_num_strides: int = 5, max_num_strides: int = 5, max_stride: int = None):
        self.max_len = max_len
        self.configs = []
        self.curr_config = []

        for i in range(min_num_strides, max_num_strides + 1):
            max_stride = max_len - i + 2 if max_stride is None else max_stride
            self.find_all_configs(self.max_len, i, max_stride)

    def find_all_configs(self, n: int, p: int, max_stride: int, config: list = None):
        """
        :param max_stride: Maximum value of stride that covers one level of trie
        :param config: current_config
        :param n: number that p integers must sum to
        :param p: number of integers that sum to n
        """

        if config is None:
            config = []

        if p == 1:
            if n <= max_stride:
                self.configs.append(config + [n])
            else:
                return
        else:
            for i in range(1, min(n - p + 2, max_stride + 1)):
                self.find_all_configs(n - i, p - 1, max_stride, config + [i])

class AllConfigGeneratorDynamic:

    def __init__(self, max_len: int = 32, num_levels: int = 5):
        self.max_len = max_len
        self.configs = []
        self.curr_config = []
        self.chosen_configs = [[]]
        self.bits_covered = []
        self.find_all_configs(max_len, num_levels)
        # for i in range(min_num_strides, max_num_strides + 1):
        #     max_stride = max_len - i + 2 if max_stride is None else max_stride
        #     self.find_all_configs(self.max_len, i, max_stride)

    def find_all_configs(self, n: int, p: int):
        """
        :param max_stride: Maximum value of stride that covers one level of trie
        :param config: current_config
        :param n: number that p integers must sum to
        :param p: number of integers that sum to n
        """
        self.chosen_configs[0] = [[x] for x in range(1, n - p + 2)]
        # For each level
        for lvl in range(1, p):
            self.chosen_configs.append([])
            # For each configuration generated in the previous level
            for config in self.chosen_configs[lvl - 1]:
                min_range = 1 if lvl < p - 1 else n - sum(config) - (p - lvl) + 1
                # Calculate all valid strides the configuration based on the amount of previous levels,
                # and the amount of bits covered so far
                for stride in range(min_range, n - sum(config) - (p - lvl) + 2):
                    self.chosen_configs[lvl].append(config + [stride])

if __name__ == '__main__':
    t1 = time.time()
    gen = BruteForceConfigGenerator(32, 8, 8)
    print('{} Configs found in {:.2f} seconds'.format(len(gen.configs), time.time() - t1))
    #configs_to_json("all_configs_from_4_to_8_levels_max_8bit.json", "all_configs_from_4_to_8_levels_max_8bit.csv", gen.configs)

    #######
    t1 = time.time()
    gen = AllConfigGeneratorDynamic(32, 8)
    print('{} Configs found in {:.2f} seconds'.format(len(gen.chosen_configs[-1]), time.time() - t1))
    pass
