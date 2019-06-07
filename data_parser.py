import pandas as pd
from glob import glob
import numpy as np
from random import shuffle


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
        df = pd.DataFrame.from_records(li).drop(0, axis=1).applymap(str)
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

        def randfor6(i):
            opt_1 = [np.random.random_integers(18, 20), np.random.random_integers(14, 17),
                     np.random.random_integers(10, 13), np.random.random_integers(4, 9),
                     np.random.random_integers(0, 3), -1]
            opt_2 = [np.random.random_integers(17, 20), np.random.random_integers(14, 16),
                     np.random.random_integers(11, 13), np.random.random_integers(4, 10),
                     np.random.random_integers(0, 3), -1]
            opt_3 = [np.random.random_integers(16, 20), np.random.random_integers(12, 15),
                     np.random.random_integers(9, 11), np.random.random_integers(3, 8),
                     np.random.random_integers(0, 2), -1]
            temp = i % 3
            if temp == 0:
                return opt_1
            elif temp == 1:
                return opt_2
            else:
                return opt_3

        def randfor7(i):
            opt_1 = [np.random.random_integers(18, 20), np.random.random_integers(14, 17),
                     np.random.random_integers(10, 13), np.random.random_integers(4, 9),
                     np.random.random_integers(2, 3), np.random.random_integers(0, 1), -1]
            opt_2 = [np.random.random_integers(17, 20), np.random.random_integers(14, 16),
                     np.random.random_integers(9, 13), np.random.random_integers(5, 8),
                     np.random.random_integers(3, 4), np.random.random_integers(0, 2), -1]
            opt_3 = [np.random.random_integers(19, 20), np.random.random_integers(15, 18),
                     np.random.random_integers(11, 14), np.random.random_integers(6, 10),
                     np.random.random_integers(3, 5), np.random.random_integers(0, 2), -1]
            temp = i % 3
            if temp == 0:
                return opt_1
            elif temp == 1:
                return opt_2
            else:
                return opt_3

        utils_dict = {}
        if num_options == 6:
            for i in range(1, self.number_of_students + 1):
                utils_dict[i] = randfor6(i)
        elif num_options == 7:
            for i in range(1, self.number_of_students + 1):
                utils_dict[i] = randfor7(i)
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
        new_dict[sid] = dict(zip(pref_df.loc[sid][:-1], utils_df.loc[sid][:-1]))
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
