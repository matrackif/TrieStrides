import json
import random
import re
import time
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


def get_empty_node_stats(tree_levels, strides):
    # Calculate number of empty child pointers in all nodes at each level
    # For example, if we have 100 nodes in level 1 (second level)
    # And the config for root node is 8, then it means the root node has 2^8 - 100 empty/null child pointers
    # TODO move this to utils.get_stats()
    num_empty_pointers_per_lvl = []
    percentage_empty_pointers_per_lvl = []
    total_num_pointers = 0
    for i in range(len(tree_levels) - 1):
        total_num_pointers_in_curr_lvl = tree_levels[i] * (2 ** strides[i])
        num_empty_pointers_in_curr_lvl = total_num_pointers_in_curr_lvl - tree_levels[i + 1]
        total_num_pointers += total_num_pointers_in_curr_lvl
        num_empty_pointers_per_lvl.append(num_empty_pointers_in_curr_lvl)
        percentage_empty_pointers_per_lvl.append(float(round(100 * num_empty_pointers_in_curr_lvl / total_num_pointers_in_curr_lvl, 2)))
    total_empty_pointer_percentage = float(round(100 * sum(num_empty_pointers_per_lvl) / total_num_pointers, 2))
    return num_empty_pointers_per_lvl, percentage_empty_pointers_per_lvl, total_empty_pointer_percentage


def get_lengths(prefixes: List[str]):
    lengths = []
    for p in prefixes:
        len_p = len(p)
        if len_p > len(lengths):
            lengths.extend([0] * (len_p - len(lengths)))
        lengths[len_p - 1] += 1
    return lengths


def configs_to_json(filename: str = "ip32_random.json", result_filename: str = "generated_configs.json", configs: list = None):
    json_as_dict = {"benchmark": "ip", "resultFile": result_filename, "dictionaries": "./../../data/ip/", "deviceId": 0,
                    "comment": "Random Configs Generated in Python", "configs": configs}

    with open(filename, 'w') as f:
        json.dump(json_as_dict, f)



