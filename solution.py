import sys

import pandas as pd


class Student:
    def __init__(self, sid: int, pref_list: list):
        self.sid = sid
        self.math_grade = None
        self.cs_grade = None
        self.pref_list = pref_list
        self.max_rejects = len(pref_list) - 1
        self.project = None
        self.top_pref = None
        self.rejects = 0

    def assign_proj(self, proj_id):
        self.project = proj_id
        free_students.discard(self.sid)

    def is_free(self):
        return bool(not self.project)

    def get_top_pref(self):
        return self.top_pref

    def set_top_pref(self):
        """Assuming {} can only appear at the end of the list of preferences"""
        if self.rejects < self.max_rejects:
            self.top_pref = self.pref_list[self.rejects]
        else:
            if self.pref_list:
                temp = self.pref_list[self.rejects]
                if type(temp) is set:
                    try:
                        self.top_pref = temp.pop()
                    except KeyError:
                        print('All possibilities depleted ! no Solution Found!')
                        sys.exit(f'The program failed with student: {self.sid}')
                else:
                    print(f'Something strange with student :{self.sid}\n check his projects pref list')
                    sys.exit(f'Something strange with student :{self.sid}\n check his projects pref list')

    def reject_student(self):
        self.project = None
        free_students.add(self.sid)
        if self.rejects < self.max_rejects:
            self.rejects += 1
        self.set_top_pref()

    def get_utility(self):
        """We define the utility to be 10/(rejects+1), unless max_rejects reached, then it's -1"""
        return 10.0 / (self.rejects + 1) if self.rejects < self.max_rejects else -1


if __name__ == '__main__':
    students_dict = {}
    free_students = set()
    df = pd.read_csv('data/train_data/students_pref_7.dat', index_col='student_id')
    for sid, pref_sr in df.iterrows():
        students_dict[sid] = Student(sid, list(pref_sr))
        free_students.add(sid)
