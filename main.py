import pandas as pd

from hw2_part1_sol import run_deferred_acceptance, run_deferred_acceptance_for_pairs, count_blocking_pairs, \
    calc_total_welfare
from hw2_part2_sol import run_market_clearing


def write_matching(matching: dict, task, n):
    df = pd.DataFrame.from_dict(matching, orient='index').sort_index()
    df.to_csv(f'matching_task_{task}_data_{n}.csv', header=['pid'], index_label='sid')
    print(f'\nmatching_task_{task}_data_{n}.csv was written\n')


def main(n):
    print(f'running for dataset {n}')

    matching_dict_1 = run_deferred_acceptance(n)
    write_matching(matching_dict_1, 'single', n)
    block_pairs_1 = count_blocking_pairs(f'matching_task_single_data_{n}.csv', n)
    welfare_1 = calc_total_welfare(f'matching_task_single_data_{n}.csv', n)
    print(f'Total welfare from the first matching: {welfare_1}')
    # print(f'Number of blocking pairs from first matching: {block_pairs_1}')

    # matching_dict_2 = run_deferred_acceptance_for_pairs(n)
    # write_matching(matching_dict_2, 'coupled', n)
    # block_pairs_2 = count_blocking_pairs(f'matching_task_coupled_data_{n}.csv', n)
    # welfare_2 = calc_total_welfare(f'matching_task_coupled_data_{n}.csv', n)
    # print(f'Total welfare from the second matching: {welfare_2}')
    # print(f'Number of blocking pairs from second matching: {block_pairs_2}')

    matching_dict_3 = run_market_clearing(n)
    write_matching(matching_dict_3, 'mcp', n)
    block_pairs_3 = count_blocking_pairs(f'matching_task_mcp_data_{n}.csv', n)
    welfare_3 = calc_total_welfare(f'matching_task_mcp_data_{n}.csv', n)
    print(f'Total welfare from the MCP matching: {welfare_3}')
    # print(f'Number of blocking pairs from MCP matching: {block_pairs_3}')


if __name__ == '__main__':
    # for i in range(1, 5):
    #     main(i)
    main('test')
    # main(3)
