import pandas as pd
from time import time
import sys
from random import randint
import timeout_decorator
import hw2_part1_sol
import hw2_part2_sol
from pathlib import PurePath, Path

from hw2_part1 import run_deferred_acceptance, run_deferred_acceptance_for_pairs, count_blocking_pairs
from hw2_part2 import run_market_clearing, calc_total_welfare


def write_matching(matching: dict, task, n):
    df = pd.DataFrame.from_dict(matching, orient='index').sort_index()
    df.to_csv(f'matching_task_{task}_data_{n}.csv', header=['pid'], index_label='sid')
    # print(f'\nmatching_task_{task}_data_{n}.csv was written\n')


def write_projects_prices(prices_dict: dict, n):
    df = pd.DataFrame.from_dict(prices_dict, orient='index').sort_index()
    df.to_csv(f'mc_task_data_{n}.csv', header=['price'], index_label='pid')
    # print(f'\nmc_task_data_{n}.csv was written\n')


def write_to_results_file(row):
    results_file = '/home/olegzendel/ec_hw2/ec_hw2/ec_2_submissions/results.csv'
    with open(results_file, 'a') as f:
        f.write(str(row).replace('[', ',').replace(']', ','))
        f.write('\n')


@timeout_decorator.timeout(60)
def run_single_matching(n):
    """Running Single Matching DA"""
    print(f'\n---------- running for dataset {n} ----------\n')
    print(f'\nTesting DA for singles')
    matching_dict_1 = run_deferred_acceptance(n)
    write_matching(matching_dict_1, 'single', n)
    block_pairs_1 = count_blocking_pairs(f'matching_task_single_data_{n}.csv', n)
    stdnts_block_pairs_1_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_single_data_{n}.csv', n)
    block_pairs_1_sol = hw2_part1_sol.count_blocking_pairs(f'../../solutions/matching_task_single_sol_data_{n}.csv', n)
    welfare_1 = calc_total_welfare(f'matching_task_single_data_{n}.csv', n)
    stdnts_welfare_1_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_single_data_{n}.csv', n)
    welfare_1_sol = hw2_part2_sol.calc_total_welfare(f'../../solutions/matching_task_single_sol_data_{n}.csv', n)
    return welfare_1, welfare_1_sol, stdnts_welfare_1_sol, block_pairs_1, block_pairs_1_sol, stdnts_block_pairs_1_sol


@timeout_decorator.timeout(60)
def test_matching(welfare, welfare_sol, stdnts_welfare_sol, block_pairs, block_pairs_sol, stdnts_block_pairs_sol):
    """Checking Matching DA"""
    welfare_calc_error = abs(welfare - stdnts_welfare_sol)
    blocking_calc_error = abs(stdnts_block_pairs_sol - block_pairs)
    welfare_matching_error = welfare_sol - stdnts_welfare_sol
    blocking_matching_error = stdnts_block_pairs_sol - block_pairs_sol
    # print(f'Welfare calc error: {welfare_calc_error}')
    # print(f'Welfare matching error: {welfare_matching_error}')
    # print(f'Total welfare from the single matching students: {stdnts_welfare_1_sol}')
    # print(f'Total welfare from the single matching solution: {welfare_1_sol}')
    # print(f'Blocking pairs calc error {blocking_calc_error}')
    # print(f'Blocking pairs matching error {blocking_matching_error}')
    # print(f'Number of blocking pairs from single matching students: {stdnts_block_pairs_1_sol}')
    # print(f'Number of blocking pairs from single matching solution: {block_pairs_1_sol}')
    return welfare_calc_error, welfare_matching_error, blocking_calc_error, blocking_matching_error


@timeout_decorator.timeout(60)
def run_paired_matching(n):
    """Running Paired matching DA"""
    print(f'\nTesting DA for pairs')
    matching_dict_2 = run_deferred_acceptance_for_pairs(n)
    write_matching(matching_dict_2, 'coupled', n)
    block_pairs_2 = count_blocking_pairs(f'matching_task_coupled_data_{n}.csv', n)
    stdnts_block_pairs_2_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_coupled_data_{n}.csv', n)
    block_pairs_2_sol = hw2_part1_sol.count_blocking_pairs(f'../../solutions/matching_task_coupled_sol_data_{n}.csv', n)
    welfare_2 = calc_total_welfare(f'matching_task_coupled_data_{n}.csv', n)
    stdnts_welfare_2_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_coupled_data_{n}.csv', n)
    welfare_2_sol = hw2_part2_sol.calc_total_welfare(f'../../solutions/matching_task_coupled_sol_data_{n}.csv', n)
    return welfare_2, welfare_2_sol, stdnts_welfare_2_sol, block_pairs_2, block_pairs_2_sol, stdnts_block_pairs_2_sol


@timeout_decorator.timeout(60)
def run_mcp(n):
    """Running Market Clearing Prices"""

    print(f'\nTesting MCP for singles')
    matching_dict_3, prices_dict = run_market_clearing(n)
    write_matching(matching_dict_3, 'mcp', n)
    write_projects_prices(prices_dict, n)
    welfare_3 = calc_total_welfare(f'matching_task_mcp_data_{n}.csv', n)
    stdnts_welfare_3_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_mcp_data_{n}.csv', n)
    welfare_3_sol = hw2_part2_sol.calc_total_welfare(f'../../solutions/matching_task_mcp_sol_data_{n}.csv', n)
    block_pairs_3 = count_blocking_pairs(f'matching_task_mcp_data_{n}.csv', n)
    stdnts_block_pairs_3_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_mcp_data_{n}.csv', n)
    block_pairs_3_sol = hw2_part1_sol.count_blocking_pairs(f'../../solutions/matching_task_mcp_sol_data_{n}.csv', n)
    return welfare_3, welfare_3_sol, stdnts_welfare_3_sol, block_pairs_3, block_pairs_3_sol, stdnts_block_pairs_3_sol


def test_pairs(n):
    """Testing that all the pairs in the matching are correct"""
    x = 1
    #TODO: Implement this Shit!


def main(n):
    temp_result = []
    try:
        results_1 = run_single_matching(n)
        temp_result.extend(test_matching(*results_1))
    except timeout_decorator.TimeoutError:
        print('TimeOut!   ....  skipping')
        temp_result.extend(['TimeOut'] * 4)
    except FileNotFoundError:
        temp_result.extend(['BadFilePath'] * 4)
    except IndexError:
        temp_result.extend(['IndexError'] * 4)
    except UnboundLocalError as error:
        temp_result.extend([error] * 4)
    except AttributeError as error:
        temp_result.extend([error] * 4)
    except KeyError as error:
        temp_result.extend([error] * 4)

    try:
        results_2 = run_paired_matching(n)
        temp_result.extend(test_matching(*results_2))
    except timeout_decorator.TimeoutError:
        print('TimeOut!   ....  skipping')
        temp_result.extend(['TimeOut'] * 4)
    except FileNotFoundError:
        temp_result.extend(['BadFilePath'] * 4)
    except IndexError:
        temp_result.extend(['IndexError'] * 4)
    except UnboundLocalError as error:
        temp_result.extend([error] * 4)
    except AttributeError as error:
        temp_result.extend([error] * 4)
    except KeyError as error:
        temp_result.extend([error] * 4)

    test = False
    try:
        results_3 = run_mcp(n)
        temp_result.extend(test_matching(*results_3))
        # print(f'Student Prices are market clearing: {test}')
        # temp_test = hw2_part2_sol.test_market_clearing(f'../../solutions/mc_task_data_{n}_sol.csv', n)
        # print(f'Solution Prices are market clearing: {temp_test}')
    except timeout_decorator.TimeoutError:
        print('TimeOut!   ....  skipping')
        temp_result.extend(['TimeOut'] * 4)
    except FileNotFoundError:
        temp_result.extend(['BadFilePath'] * 4)
    except IndexError:
        temp_result.extend(['IndexError'] * 4)
    except UnboundLocalError as error:
        temp_result.extend([error] * 4)
    except AttributeError as error:
        temp_result.extend([error] * 4)
    except KeyError as error:
        temp_result.extend([error] * 4)
    else:
        try:
            test = hw2_part2_sol.test_market_clearing(f'mc_task_data_{n}.csv', n)
        except KeyError:
            test = 'MissingPID'

    result_list.extend([f'data set {n}'] + temp_result + [test])


if __name__ == '__main__':
    cwd = Path.cwd().name
    ids = cwd.strip('hw2').split('_')[1:]
    print(f'\nChecking {ids}')
    if len(ids) < 2:
        ids.append('000000000')
    result_list = ids
    for i in range(1, 5):
        x = time()
        try:
            main(i)
        except TypeError:
            main(str(i))
        print(f'\nRunning time: {time() - x} sec')
    write_to_results_file(result_list)
