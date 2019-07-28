import pandas as pd
from time import time
from random import randint
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


def main(n):
    print(f'\n---------- running for dataset {n} ----------\n')
    print(f'\nTesting DA for singles')
    matching_dict_1 = run_deferred_acceptance(n)
    matching_dict_1_sol = hw2_part1_sol.run_deferred_acceptance(n)
    write_matching(matching_dict_1, 'single', n)
    write_matching(matching_dict_1_sol, 'single_sol', n)
    block_pairs_1 = count_blocking_pairs(f'matching_task_single_data_{n}.csv', n)
    stdnts_block_pairs_1_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_single_data_{n}.csv', n)
    block_pairs_1_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_single_sol_data_{n}.csv', n)
    welfare_1 = calc_total_welfare(f'matching_task_single_data_{n}.csv', n)
    stdnts_welfare_1_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_single_data_{n}.csv', n)
    welfare_1_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_single_sol_data_{n}.csv', n)
    ############
    welfare_1_calc_error = abs(welfare_1 - stdnts_welfare_1_sol)
    blocking_1_calc_error = abs(stdnts_block_pairs_1_sol - block_pairs_1)
    welfare_1_matching_error = welfare_1_sol - stdnts_welfare_1_sol
    blocking_1_matching_error = stdnts_block_pairs_1_sol - block_pairs_1_sol
    print(f'Welfare calc error: {welfare_1_calc_error}')
    print(f'Welfare matching error: {welfare_1_matching_error}')
    # print(f'Total welfare from the single matching students: {stdnts_welfare_1_sol}')
    # print(f'Total welfare from the single matching solution: {welfare_1_sol}')
    print(f'Blocking pairs calc error {blocking_1_calc_error}')
    print(f'Blocking pairs matching error {blocking_1_matching_error}')
    # print(f'Number of blocking pairs from single matching students: {stdnts_block_pairs_1_sol}')
    # print(f'Number of blocking pairs from single matching solution: {block_pairs_1_sol}')

    print(f'\nTesting DA for pairs')
    matching_dict_2 = run_deferred_acceptance_for_pairs(n)
    matching_dict_2_sol = hw2_part1_sol.run_deferred_acceptance_for_pairs(n)
    write_matching(matching_dict_2, 'coupled', n)
    write_matching(matching_dict_2_sol, 'coupled_sol', n)
    block_pairs_2 = count_blocking_pairs(f'matching_task_coupled_data_{n}.csv', n)
    stdnts_block_pairs_2_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_coupled_data_{n}.csv', n)
    block_pairs_2_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_coupled_sol_data_{n}.csv', n)
    welfare_2 = calc_total_welfare(f'matching_task_coupled_data_{n}.csv', n)
    stdnts_welfare_2_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_coupled_data_{n}.csv', n)
    welfare_2_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_coupled_sol_data_{n}.csv', n)
    ############
    welfare_2_calc_error = abs(welfare_2 - stdnts_welfare_2_sol)
    blocking_2_calc_error = abs(stdnts_block_pairs_2_sol - block_pairs_2)
    welfare_2_matching_error = welfare_2_sol - stdnts_welfare_2_sol
    blocking_2_matching_error = stdnts_block_pairs_2_sol - block_pairs_2_sol
    print(f'Welfare calc error: {welfare_2_calc_error}')
    print(f'Welfare matching error: {welfare_2_matching_error}')
    # print(f'Total welfare from the coupled matching students: {stdnts_welfare_2_sol}')
    # print(f'Total welfare from the coupled matching solution: {welfare_2_sol}')
    print(f'Blocking pairs calc error {blocking_2_calc_error}')
    print(f'Blocking pairs matching error {blocking_2_matching_error}')
    # print(f'Number of blocking pairs from coupled matching students: {stdnts_block_pairs_2_sol}')
    # print(f'Number of blocking pairs from coupled matching solution: {block_pairs_2_sol}')

    print(f'\nTesting MCP for singles')
    matching_dict_3, prices_dict = run_market_clearing(n)
    print('Running solution - student finished')
    matching_dict_3_sol, prices_dict_sol = hw2_part2_sol.run_market_clearing(n)
    write_matching(matching_dict_3, 'mcp', n)
    write_matching(matching_dict_3_sol, 'mcp_sol', n)
    write_projects_prices(prices_dict, n)
    write_projects_prices(prices_dict_sol, f'{n}_sol')
    welfare_3 = calc_total_welfare(f'matching_task_mcp_data_{n}.csv', n)
    stdnts_welfare_3_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_mcp_data_{n}.csv', n)
    welfare_3_sol = hw2_part2_sol.calc_total_welfare(f'matching_task_mcp_sol_data_{n}.csv', n)
    block_pairs_3 = count_blocking_pairs(f'matching_task_mcp_data_{n}.csv', n)
    stdnts_block_pairs_3_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_mcp_data_{n}.csv', n)
    block_pairs_3_sol = hw2_part1_sol.count_blocking_pairs(f'matching_task_mcp_sol_data_{n}.csv', n)
    ############
    welfare_3_calc_error = abs(welfare_3 - stdnts_welfare_3_sol)
    welfare_3_matching_error = welfare_3_sol - stdnts_welfare_3_sol
    blocking_3_calc_error = abs(stdnts_block_pairs_3_sol - block_pairs_3)
    blocking_3_matching_error = stdnts_block_pairs_3_sol - block_pairs_3_sol
    print(f'Welfare calc error: {welfare_3_calc_error}')
    print(f'Welfare matching error: {welfare_3_matching_error}')
    # print(f'Total welfare from the MCP matching student: {stdnts_welfare_3_sol}')
    # print(f'Total welfare from the MCP matching solution: {welfare_3_sol}')
    print(f'Blocking pairs calc error {blocking_3_calc_error}')
    print(f'Blocking pairs matching error {blocking_3_matching_error}')
    # print(f'Number of blocking pairs from MCP matching student: {stdnts_block_pairs_3_sol}')
    # print(f'Number of blocking pairs from MCP matching: {block_pairs_3_sol}')


if __name__ == '__main__':
    print(Path().cwd())
    exit()
    for i in range(1, 5):
        x = time()
        main(i)
        print(f'\nRunning time: {time() - x} sec')
    # x = time()
    # main(5)
    # print(f'Running time for dataset 5: {time() - x} sec')
    # x = time()
    # main(3)
    # print(f'Running time for dataset 3: {time() - x} sec')
    # x = time()
    # main(2)
    # print(f'Running time for dataset 2: {time() - x} sec')
    # x = time()
    # main(4)
    # print(f'Running time for dataset 4: {time() - x} sec')
    # x = time()
    # main(1)
    # print(f'Running time for dataset 1: {time() - x} sec')
    # main(2)
    # main('test')
    # main('test_mc')
