import re
import time
import random
import json
from typing import List

DEBUG = False

def print_d(x: str):
    if DEBUG:
        print(x)


def binary_to_int(binary_num: str):
    num_as_int = 0
    exponent = 0
    for i in range(len(binary_num) - 1, -1, -1):
        if binary_num[i] == '1':
            num_as_int += (2 ** exponent)
        exponent += 1
    return num_as_int


def int_to_binary_str(num: int, bit_length: int = 8) -> str:
    binary_str = ['0'] * bit_length
    idx = bit_length - 1

    while num > 0:
        binary_str[idx] = str(num % 2)
        idx -= 1
        num = int(num / 2)

    return "".join(binary_str)


def remove_leading_zeroes(binary_str: str) -> str:
    for i in range(len(binary_str)):
        if binary_str[i] == '1':
            return binary_str[i:]
    return binary_str


def get_cost_of_1bit_trie(nodes: List[int], ignore_last_level: bool = False):
    one_bit_trie_sum = 0
    num_levels = len(nodes) - 1 if ignore_last_level else len(nodes)
    for i in range(num_levels):
        one_bit_trie_sum += 2 * nodes[i]
    return one_bit_trie_sum


def get_cost_of_trie(nodes: List[int], strides: List[int], ignore_last_level: bool = False):
    cost = 0
    stride_sum = 0
    num_nodes = []
    num_levels = len(strides) - 1 if ignore_last_level else len(strides)
    for i in range(num_levels):
        if stride_sum > len(nodes) - 1:
            print('Attempting to find cost at level:', stride_sum, 'when len(nodes) is: ', len(nodes))
            cost_at_curr_level = 0
        else:
            cost_at_curr_level = nodes[stride_sum] * (2 ** strides[i])
        num_nodes.append(cost_at_curr_level)
        cost += cost_at_curr_level
        stride_sum += strides[i]
    return cost, num_nodes


class Node:
    def __init__(self, left_child=None, right_child=None):
        self.right_child = right_child
        self.left_child = left_child


class Tree:
    def __init__(self, prefixes: List[str]):
        self.root = Node()
        max_len = max(len(p) for p in prefixes)
        self.node_counts = [0] * max_len
        self.node_counts[0] += 1
        for prefix in prefixes:
            self.insert(prefix)

    def insert(self, key: str):
        curr_node = self.root
        for i in range(1, len(key)):
            b = int(key[i - 1])
            if b == 0:
                if curr_node.left_child is None:
                    curr_node.left_child = Node()
                    self.node_counts[i] += 1
                curr_node = curr_node.left_child
            else:
                if curr_node.right_child is None:
                    curr_node.right_child = Node()
                    self.node_counts[i] += 1
                curr_node = curr_node.right_child


def get_node_counts(prefixes: List[str]) -> List[int]:
    return Tree(prefixes).node_counts


def get_file_lines(file_name: str) -> List[str]:
    with open(file_name) as f:
        return f.readlines()


def get_prefixes_from_file(file_name: str = 'data/data-raw-table_australia_012016.txt') -> List[str]:
    start_time = time.time()
    lines = get_file_lines(file_name)
    prefixes = []
    for line in lines:
        # print(line)
        split_line = list(filter(None, re.split("[;.\n]+", line)))
        if len(split_line) != 7:
            print('ERROR: Line length is not 7! Line:', split_line)
            continue
        else:
            prefix = ""
            for num in split_line[2:6]:
                prefix += int_to_binary_str(int(num))
            prefix = prefix[:int(split_line[6])]  # Remove trailing zeroes, leave only significant bits used for mask
            # prefix = remove_leading_zeroes(prefix)
            prefixes.append(prefix)
    print('Read', len(prefixes), 'prefixes from file in %s seconds' % (time.time() - start_time))
    return prefixes


def get_stats(nodes, strides, ignore_last_level: bool = False, print_results: bool = False):
    len_nodes = len(nodes)
    # if len_nodes != bits_covered:
    #     print("Trie covers", len_nodes, "bits but strides:", strides, "cover", bits_covered, "bits")
    #     return
    one_bit_trie_sum = get_cost_of_1bit_trie(nodes, ignore_last_level)
    cost, strides_nodes = get_cost_of_trie(nodes, strides, ignore_last_level)
    diff = one_bit_trie_sum - cost
    percent = (cost / one_bit_trie_sum) * 100.0
    if print_results:
        print('Maximum key length:', len_nodes, '\nInput strides:', strides,
              '\nNumber of nodes at each level of 1-bit trie:', nodes,
              '\nNumber of nodes in each level of strides trie:',
              strides_nodes, '\nMax number of nodes at level:', max(strides_nodes),
              '\nTotal number of units needed in 1 bit trie:', one_bit_trie_sum,
              '\nTotal number of units needed in strides trie:',
              cost, '\nSaved', diff, 'nodes')
        print('Strides trie is: {}% the size of 1 bit trie'.format(percent))
    return str(nodes), str(strides_nodes), cost, percent


def get_lengths(prefixes: List[str]):
    lengths = []
    for p in prefixes:
        len_p = len(p)
        if len_p > len(lengths):
            lengths.extend([0] * (len_p - len(lengths)))
        lengths[len_p - 1] += 1
    return lengths


class ConfigurationGenerator(random.Random):

    def __init__(self, seed_val=0, max_len=1):
        # Using the name seed_value, in order for it not to be confused with the seed() method in Random class
        super().__init__(seed_val)
        self.seed_value = seed_val
        self.max_len = max_len
        self.configs = {}

    def gen_config(self, num_levels: int):
        if num_levels == 1:
            return [self.max_len]
        elif num_levels == self.max_len:
            return [1] * self.max_len

        config = []
        levels_remaining = num_levels
        end_range = self.max_len - levels_remaining + 1
        # The intervals for possible stride values get smaller and smaller
        # This often means that the end of the configuration is full of smaller numbers
        # In order to mitigate this effect, we randomize the interval we choose the random number from
        end_range = self.randint(1, end_range)
        bits_to_cover = self.max_len
        while levels_remaining > 0:
            stride = 0
            if levels_remaining > 1:
                stride = super().randint(1, end_range)
            elif levels_remaining == 1:
                stride = end_range
            config.append(stride)
            bits_to_cover -= stride
            levels_remaining -= 1
            end_range = bits_to_cover - levels_remaining + 1
            end_range = self.randint(1, end_range)
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


def random_configs_to_json(filename: str = "ip32_random.json", min_num_levels: int = 3, max_num_levels: int = 10, num_configs_per_level: int = 100, max_len: int = 32, seed: int = 0):
    # TODO spread out the config to prevent all ones at the end
    json_as_dict = {"benchmark": "ip",
                    "resultFile": "ip32_random_results.csv",
                    "dictionaries": "./../../data/ip/",
                    "deviceId": 0,
                    "comment": "Random Configs Generated in Python"}

    c = ConfigurationGenerator(seed_val=seed, max_len=max_len)
    configs = c.gen_configs(min_num_levels, max_num_levels, num_configs_per_level)
    configs_as_list_of_lists = []
    for num_strides in configs.keys():
        for config in configs[num_strides]:
            configs_as_list_of_lists.append(config)
    json_as_dict["configs"] = configs_as_list_of_lists

    with open(filename, 'w') as f:
        json.dump(json_as_dict, f)
