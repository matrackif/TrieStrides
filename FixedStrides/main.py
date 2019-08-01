from typing import List
from copy import deepcopy
import numpy as np
import utils.utils as utils
import time
import argparse
import itertools
DEBUG = False


def print_d(x: str):
    if DEBUG:
        print(x)


def get_stats(nodes, strides):
    len_nodes = len(nodes)
    bits_covered = np.sum(strides)
    # if len_nodes != bits_covered:
    #     print("Trie covers", len_nodes, "bits but strides:", strides, "cover", bits_covered, "bits")
    #     return
    one_bit_trie_sum = utils.get_cost_of_1bit_trie(nodes)
    cost, strides_nodes = utils.get_cost_of_trie(nodes, strides)
    diff = one_bit_trie_sum - cost
    percent = (cost / one_bit_trie_sum) * 100.0
    print('Maximum key length:', len_nodes, '\nInput strides:', strides,
          '\nNumber of nodes at each level of 1-bit trie:', nodes,
          '\nNumber of nodes in each level of strides trie:',
          strides_nodes, '\nTotal number of units needed in 1 bit trie:', one_bit_trie_sum,
          '\nTotal number of units needed in strides trie:',
          cost, '\nSaved', diff, 'nodes')
    print('Strides trie is: {}% the size of 1 bit trie'.format(percent))
    return str(nodes), str(strides_nodes), cost, percent


# From the Wiley document by Wolant and Kaczmarski
# TODO verify this way also works and create RC array
def get_node_counts2(prefixes: List[str]):
    prefixes = sorted(prefixes, key=utils.binary_to_int)
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


def fixed_strides(prefixes: List[str]):
    max_len = max(len(p) for p in prefixes)
    k = max_len
    c = {}
    m = {}
    nodes = utils.get_node_counts(prefixes)

    c[-1] = [0] * k
    for i in range(max_len):
        c[i] = [(2 ** (i + 1)) if j == 1 else 0 for j in range(max_len + 1)]
        m[i] = [-1 if j == 1 else 0 for j in range(max_len + 1)]

    print_d('nodes: ' + str(nodes) + ' c:' + str(c) + '\n' ' m:' + str(m))
    for r in range(2, k + 1):
        for j in range(r - 1, max_len):
            min_j = max(m[j - 1][r], m[j][r - 1])
            print_d('m: ' + str(m) + 'min_j: ' + str(min_j))
            print_d('c: ' + str(c))
            min_cost = c[j][r - 1]
            min_l = m[j][r - 1]
            for z in range(min_j, j):
                print_d('r: ' + str(r) + ' j: ' + str(j) + ' z: ' + str(z))
                cost = c[z][j - 1] + (nodes[z + 1] * (2 ** (j - z)))
                if cost < min_cost:
                    min_cost = cost
                    min_l = z
            c[j][r] = min_cost
            m[j][r] = min_l
    print_d('------------------------')
    print_d('c:' + str(c) + '\n' ' m:' + str(m))
    ############################################

    strides = []
    levels_to_cover = max_len - 1  # cover levels 0 through "levels_to_cover" (levels_to_cover + 1 = max_len)
    tmp = 0
    while levels_to_cover >= 0:
        min_m = m[levels_to_cover][levels_to_cover + 1]
        tmp += c[levels_to_cover][levels_to_cover + 1]
        stride = levels_to_cover - min_m
        levels_to_cover -= stride
        strides.append(stride)

    strides = strides[-1::-1]  # reverse strides array
    # print('c[max_len - 1][max_len]:', c[max_len - 1][max_len])
    return strides, nodes


def fixed_strides_2(prefixes: List[str], k: int = None):
    max_len = max(len(p) for p in prefixes)
    # k = max_len
    c = {}
    m = {}
    nodes = utils.get_node_counts(prefixes)

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


def get_prefixes_from_file(file_name: str):
    start_time = time.time()
    prefixes = utils.get_binary_prefixes_from_file(file_name=file_name)
    print('Read', len(prefixes), 'prefixes from file in %s seconds' % (time.time() - start_time))
    return prefixes

"""
class unique_element:
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences


def perm_unique(elements):
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)


def perm_unique_helper(listunique,result_list,d):
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1


def unique_permutations(seq):
  
    # Yield only unique permutations of seq in an efficient way.
    # 
    # A python implementation of Knuth's "Algorithm L", also known from the
    # std::next_permutation function of C++, and as the permutation algorithm
    # of Narayana Pandita.
   

    # Precalculate the indices we'll be iterating over for speed
    i_indices = range(len(seq) - 1, -1, -1)
    k_indices = i_indices[1:]

    # The algorithm specifies to start with a sorted version
    seq = sorted(seq)

    while True:
        yield seq

        # Working backwards from the last-but-one index,           k
        # we find the index of the first decrease in value.  0 0 1 0 1 1 1 0
        for k in k_indices:
            if seq[k] < seq[k + 1]:
                break
        else:
            # Introducing the slightly unknown python for-else syntax:
            # else is executed only if the break statement was never reached.
            # If this is the case, seq is weakly decreasing, and we're done.
            return

        # Get item from sequence only once, for speed
        k_val = seq[k]

        # Working backwards starting with the last item,           k     i
        # find the first one greater than the one at k       0 0 1 0 1 1 1 0
        for i in i_indices:
            if k_val < seq[i]:
                break

        # Swap them in the most efficient way
        (seq[k], seq[i]) = (seq[i], seq[k])                #       k     i
                                                           # 0 0 1 1 1 1 0 0

        # Reverse the part after but not                           k
        # including k, also efficiently.                     0 0 1 1 0 0 1 1
        seq[k + 1:] = seq[-1:k:-1]


def brute_force(prefixes: List[str]):
    nodes = utils.get_node_counts(prefixes)
    max_len = max(len(p) for p in prefixes)
    l = []
    for i in range(1, max_len + 1):
        l.extend([i] * int(max_len / i))
    # results = [list(seq) for i in range(len(l), 0, -1) for seq in itertools.combinations(l, i) if sum(seq) == max_len]
    results = []
    for gg in unique_permutations(l):
        if sum(gg) == max_len:
            results.append(gg)

    min_cost = (2 ** 63) - 1
    min_nodes = []
    min_strides = []
    for strides in results:
        cost, strides_nodes = utils.get_cost_of_trie(nodes, strides)
        if cost < min_cost:
            min_cost = cost
            min_nodes = strides_nodes
            min_strides = strides
    get_stats(nodes, min_strides)
    return min_cost
"""


def run_algo(prefixes: List[str]):
    start_time = time.time()

    strides, nodes = fixed_strides_2(prefixes)
    end_time = time.time() - start_time
    print('Strides: %s ' % strides)
    print(
        'Strides found from file in %s seconds. R-Trie covers %s bits' % (end_time, np.sum(strides)))
    get_stats(nodes, strides)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', nargs='?', type=str, default='data/data-raw-table_australia_012016.txt',
                        const='data/data-raw-table_australia_012016.txt', help='Path to IP prefix file')
    args = vars(parser.parse_args())
    """
    run_algo(['000', '001', '010', '011', '100', '101', '110', '111'])
    run_algo(['0000', '0001', '0010', '0011', '0100', '0101', '0110', '1000', '1001', '1010', '1011', '1100', '1101', '1110', '1111'])
    run_algo(['000', '0', '01', '100', '001'])
    run_algo(['1', '0', '00'])
    run_algo(['0', '1', '10', '11', '110', '001', '111', '000', '101'])
    run_algo(['0', '1', '10', '11', '110', '001', '111', '000', '101', '1000', '0000'])
    run_algo(['000', '001'])
    run_algo(['000', '001', '0', '100'])
    run_algo(['0', '1'])
    run_algo(['1111111111'])
    run_algo(['11111', '00000'])
    run_algo(['00', '10'])
    run_algo(['00', '01'])
    run_algo(test)
    run_algo(example_from_doc)
    run_algo(['000', '001', '010', '011', '100', '101', '110', '111'])
    run_algo(['11111', '00000'])
    brute_force(example_from_doc)
    """
    # example_from_doc = ['00', '01', '10', '11', '11100', '11101', '11110', '11111', '11001', '10000', '10001', '1000001', '1000000']
    test = ['0', '1', '00', '11', '10', '01']
    prefixes = get_prefixes_from_file(file_name=args['file'])
    run_algo(prefixes)
