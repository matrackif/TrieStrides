import argparse
import time
from typing import List

import numpy as np

import algorithms.algos as algos
import utils.utils as utils


def run_algo(prefixes: List[str]):
    start_time = time.time()

    strides, nodes = algos.fixed_strides_2(prefixes)
    end_time = time.time() - start_time
    print('Strides: %s ' % strides)
    print(
        'Fixed strides algorithm completed in %s seconds. R-Trie covers %s bits' % (end_time, np.sum(strides)))
    utils.get_stats(nodes, strides, False, True)


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
    dense_tree_at_very_end = ['00000000', '00000001', '00000010', '00000011', '00000100', '00000101', '00000110', '00000111', '10', '11']
    example_from_second_doc = ['0', '11', '110', '1110', '11000', '11111', '1101010']
    # prefixes = utils.get_prefixes_from_file(file_name=args['file'])
    run_algo(example_from_second_doc)
    algos.get_max_mem_per_level(example_from_second_doc)
