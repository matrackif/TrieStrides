from typing import List
import math
import re


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


def get_cost_of_1bit_trie(nodes: List[int]):
    one_bit_trie_sum = 0
    for i in range(len(nodes)):
        one_bit_trie_sum += 2 * nodes[i]
    return one_bit_trie_sum


def get_cost_of_trie(nodes: List[int], strides: List[int]):
    cost = 0
    stride_sum = 0
    num_nodes = []
    for i in range(len(strides)):
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


def get_file_lines(file_name: str) ->List[str]:
    with open(file_name) as f:
        return f.readlines()


def get_binary_prefixes_from_file(file_name: str = 'data/data-raw-table_australia_012016.txt') -> List[str]:
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
    return prefixes


