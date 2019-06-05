import pandas as pd
from glob import glob
import numpy as np


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

    def write_df(self, df: pd.DataFrame, dat_type):
        df.to_csv(f'./data/train_data/{dat_type}_{self.file_name.replace("toc", "dat")}')


def read_files():
    toc_files = glob('./data/train_data/*.toc')
    # toc_files = glob('./data/*.soi')
    for toc_file in toc_files:
        parser = DataParser(toc_file)
        parser.write_df(parser.gen_grades_dat(), 'grades')
        parser.write_df(parser.pref_df, 'students_pref')
        

if __name__ == '__main__':
    read_files()
