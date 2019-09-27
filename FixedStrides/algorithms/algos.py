from copy import deepcopy
from typing import List, Dict, Tuple

import numpy as np

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
    # k = max_len
    c = {}
    m = {}
    nodes = get_node_counts(prefixes)

    if k is None:
        k = max_len

    c[-1] = [0] * k
    for i in range(max_len):
        c[i] = [(2 ** (i + 1)) if j == 1 else 0 for j in range(max_len + 1)]
        m[i] = [-1 if j == 1 else 0 for j in range(max_len + 1)]

    print_d('nodes: ' + str(nodes) + ' c:' + str(c) + '\n' ' m:' + str(m))
    for j in range(1, max_len):
        for r in range(2, k + 1):
            min_j = max(m[j - 1][r], m[j][r - 1])
            print_d('m: ' + str(m) + 'min_j: ' + str(min_j))
            print_d('c: ' + str(c))
            min_cost = c[j][r - 1]
            min_l = m[j][r - 1]
            for z in range(min_j, j):
                print_d('r: ' + str(r) + ' j: ' + str(j) + ' z: ' + str(z))
                cost = c[z][j - 1] + (nodes[z + 1] * (2 ** (j - z)))
                if cost < min_cost:
                    # print('New min cost found, r:', r, 'j:', j, 'z:', z, 'old cost:', min_cost, 'new cost:', cost)
                    min_cost = cost
                    min_l = z
            c[j][r] = min_cost
            m[j][r] = min_l
    print_d('------------------------')
    print_d('c:' + str(c) + '\n' ' m:' + str(m))
    ############################################

    strides = []
    levels_to_cover = max_len - 1  # cover levels 0 through "levels_to_cover" (levels_to_cover + 1 = max_len)
    tmp_k = k
    while levels_to_cover >= 0:
        min_m = m[levels_to_cover][tmp_k]
        # tmp += c[levels_to_cover][levels_to_cover + 1]
        stride = levels_to_cover - min_m
        tmp_k -= stride
        levels_to_cover -= stride
        strides.append(stride)

    strides = strides[-1::-1]  # reverse strides array
    # print('c[max_len - 1][max_len]:', c[max_len - 1][max_len])
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
    print('distribute_prefixes() found intervals:', intervals)
    # TODO resolve overlapping intervals by making all intervals disjoint
    # TODO In regions of overlapping intervals, make sure to give priority to interval covering most prefixes
    strides = []
    return strides

