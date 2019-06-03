import pandas as pd
from glob import glob
from csv import reader


def toc_file_parser(toc_file):
    with open(toc_file) as file:
        reader_ = list(reader(file))
        projects_len = int(reader_[0][0])
        print(projects_len)
        for row in reader_[projects_len + 2:]:
            print(row)
            exit()


def read_files():
    toc_files = glob('./data/train_data/*.toc')
    for toc_file in toc_files:
        toc_file_parser(toc_file)
    print(toc_files)


if __name__ == '__main__':
    read_files()
