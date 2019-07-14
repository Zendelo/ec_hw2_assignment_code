import pandas as pd
from glob import glob
import numpy as np
from random import shuffle, sample, choice


class DataParser:
    def __init__(self, data_file):
        self.file = data_file
        self.file_name = data_file.rsplit('\\')[-1]  # For Windows
        # self.file_name = data_file.rsplit('/')[-1]  # For Linux
        self.number_of_projects = None  # will be set by the parsing method
        self.pref_df = self.pref_file_parser()
        self.number_of_students = len(self.pref_df)
        self.gen_grades_dat()

    def pref_file_parser(self):
        li = []
        with open(self.file) as file:
            projects_len = int(file.readline())
            self.number_of_projects = projects_len
            print(projects_len)
            for i, line in enumerate(file):
                if i > projects_len:
                    li.append(eval(f'[{line}]'))
        df = pd.DataFrame.from_records(li).drop(0, axis=1)
        df = df.drop(len(df.columns), axis=1)
        uniq_projects = set()
        for row in df.values:
            uniq_projects.update(set(row))
        rename_pid = {}
        for i, pid in enumerate(uniq_projects):
            rename_pid[pid] = i
        df = df.applymap(lambda x: rename_pid[x])
        df = df.applymap(lambda x: x % len(df)).applymap(str)
        df.insert(loc=0, column='student_id', value=range(1, len(df) + 1))
        return df.set_index('student_id')

    def gen_grades_dat(self):
        math_grades = np.random.normal(loc=78, scale=10, size=self.number_of_students)
        cs_grades = np.random.normal(loc=82, scale=10, size=self.number_of_students)
        df = pd.DataFrame({'math_grades': math_grades, 'cs_grades': cs_grades},
                          index=range(1, self.number_of_students + 1)).clip(56, 100)
        df.index.name = 'student_id'
        return df.round(2)

    def gen_pairs_dat(self):
        student_ids = list(range(1, self.number_of_students + 1))
        shuffle(student_ids)
        middle = int(len(student_ids) / 2)
        if len(student_ids[:middle]) == len(student_ids[middle:]):
            df = pd.DataFrame({'partner_1': student_ids[middle:], 'partner_2': student_ids[:middle]})
        else:
            df = pd.DataFrame({'partner_1': student_ids[middle:], 'partner_2': student_ids[:middle] + ['']})
        return df

    def gen_util_dat(self):
        num_options = len(self.pref_df.columns)

        def randfor5(i):
            opt_1 = [np.random.randint(18, 21), np.random.randint(14, 18), np.random.randint(10, 14),
                     np.random.randint(4, 10), np.random.randint(-1, 4)]
            opt_2 = [np.random.randint(17, 21), np.random.randint(14, 17), np.random.randint(11, 14),
                     np.random.randint(4, 11), np.random.randint(-1, 4)]
            opt_3 = [np.random.randint(16, 21), np.random.randint(12, 16), np.random.randint(9, 12),
                     np.random.randint(3, 9), np.random.randint(-1, 3)]
            temp = i % 3
            if temp == 0:
                return opt_1
            elif temp == 1:
                return opt_2
            else:
                return opt_3

        def randfor6(i):
            opt_1 = [np.random.randint(18, 21), np.random.randint(14, 18), np.random.randint(10, 14),
                     np.random.randint(4, 10), np.random.randint(2, 4), np.random.randint(-1, 1)]
            opt_2 = [np.random.randint(17, 21), np.random.randint(14, 17), np.random.randint(9, 14),
                     np.random.randint(5, 9), np.random.randint(3, 5), np.random.randint(-1, 3)]
            opt_3 = [np.random.randint(19, 21), np.random.randint(15, 19), np.random.randint(11, 15),
                     np.random.randint(6, 11), np.random.randint(3, 6), np.random.randint(-1, 3)]
            temp = i % 3
            if temp == 0:
                return opt_1
            elif temp == 1:
                return opt_2
            else:
                return opt_3

        utils_dict = {}
        if num_options == 5:
            for i in range(1, self.number_of_students + 1):
                utils_dict[i] = randfor5(i)
        elif num_options == 6:
            for i in range(1, self.number_of_students + 1):
                utils_dict[i] = randfor6(i)
        else:
            print('Error!')
            exit()
        df = pd.DataFrame.from_dict(utils_dict, orient='index')
        df.columns = self.pref_df.columns
        df.index.name = 'student_id'
        return df

    def write_df(self, df: pd.DataFrame, dat_type):
        if dat_type == 'pairs':
            index = False
        else:
            index = True
        df.to_csv(f'./data/train_data/{dat_type}_{self.file_name.replace("toc", "csv")}', index=index)
        # print(df)


def read_files():
    toc_files = glob('./data/train_data/*.toc')
    # toc_files = glob('./data/*.soi')
    for toc_file in toc_files:
        parser = DataParser(toc_file)
        parser.write_df(parser.gen_util_dat(), 'util')
        parser.write_df(parser.gen_pairs_dat(), 'pairs')
        parser.write_df(parser.gen_grades_dat(), 'grades')
        parser.write_df(parser.pref_df, 'students_pref')
        big_df = merge_data_files(toc_file)
        parser.write_df(big_df, 'preferences')


def merge_data_files(toc_file):
    n = toc_file.split('\\')[-1].split('.')[0]
    # n = toc_file.split('/')[-1].split('.')[0]
    pref_df = pd.read_csv(f'data/train_data/students_pref_{n}.csv', index_col='student_id')
    utils_df = pd.read_csv(f'data/train_data/util_{n}.csv', index_col='student_id')
    new_dict = {}
    for sid in pref_df.index:
        new_dict[sid] = dict(zip(pref_df.loc[sid][:], utils_df.loc[sid][:]))
    df = pd.DataFrame.from_dict(new_dict, orient='index').fillna(-1).astype(int)
    df = df.reindex(sorted(df.columns), axis=1)
    df.index.name = 'student_id'
    return df


def testing(file):
    parser = DataParser(file)
    print(parser.pref_df.describe())


if __name__ == '__main__':
    read_files()
    # merge_data_files()
    # testing('data/train_data/4.toc')
