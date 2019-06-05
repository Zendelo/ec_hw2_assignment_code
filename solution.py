import sys

import pandas as pd
import numpy as np


class Student:
    free_students = set()

    def __init__(self, sid: int, pref_list: list, math_grade, cs_grade, utils):
        self.sid = sid
        self.math_grade = math_grade
        self.cs_grade = cs_grade
        self.pref_list = pref_list
        self.max_rejects = len(pref_list) - 1
        self.rejects = 0
        self.project = None
        self.top_pref = None
        self.set_top_pref()
        self.utils = utils
        self.pair = None
        self.main_partner = True

    def assign_proj(self, proj_id):
        self.project = proj_id
        Student.free_students.discard(self.sid)
        if self.pair:
            self.pair.project = proj_id

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
        Student.free_students.add(self.sid)
        if self.rejects < self.max_rejects:
            self.rejects += 1
        self.set_top_pref()

    def get_utility(self):
        if self.main_partner:
            return self.utils[self.rejects]
        else:
            try:
                idx = self.pref_list.index(self.project.pid)
            except ValueError:
                idx = self.max_rejects
            return self.utils[idx]


class Project:
    def __init__(self, pid):
        self.pid = pid
        self.grade_type = 'cs_grade' if pid % 2 else 'math_grade'
        self.proposals = {}
        self.student = None

    def is_free(self):
        return bool(not self.student)

    def accept_offer(self, student):
        if self.student:
            self.student.reject_student()
        self.student = student

    def test_offer(self, nominee):
        if getattr(nominee, self.grade_type) > getattr(self.student, self.grade_type):
            self.accept_offer(nominee)

    def add_offer(self, student):
        self.proposals[student.sid] = getattr(student, self.grade_type)


def create_students(df, grades_df, utils_df):
    students_dict = {}
    for sid, pref_sr in df.iterrows():
        students_dict[sid] = Student(sid, list(pref_sr), grades_df.loc[sid].math_grades, grades_df.loc[sid].cs_grades,
                                     utils_df.loc[sid])
        Student.free_students.add(sid)
    return students_dict


def create_projects(df):
    projects_dict = {}
    uniq_projects = pd.unique(df[df.columns[:-1]].values.ravel('K'))
    for project_id in uniq_projects:
        projects_dict[project_id] = Project(project_id)
    return projects_dict


def make_offers(students_dict, projects_dict):
    for sid in Student.free_students:
        student = students_dict[sid]
        project = projects_dict[student.get_top_pref()]
        project.add_offer(student)


def respond_offers(students_dict, projects_dict):
    for pid, project in projects_dict.items():
        if not project.proposals:
            continue
        top_sid = max(project.proposals, key=(lambda key: project.proposals[key]))
        project.proposals.pop(top_sid)
        for rej_sid in project.proposals:
            students_dict[rej_sid].reject_student()
        project.proposals = {}
        students_dict[top_sid].assign_proj(project)


def deferred_acceptance(students_dict, projects_dict):
    while Student.free_students:
        make_offers(students_dict, projects_dict)
        respond_offers(students_dict, projects_dict)


def task_one(pref_df, grades_df, util_df):
    students_dict = create_students(pref_df, grades_df, util_df)
    projects_dict = create_projects(pref_df)
    deferred_acceptance(students_dict, projects_dict)
    total_welfare = 0
    for sid, student in students_dict.items():
        if student.is_free():
            print('wtf!')
            sys.exit(sid)
        total_welfare += student.get_utility()
        # print(f'Student {sid} -> {student.project.pid}')
        # print(f'utility: {student.get_utility() :.2f}')
        # print(f'rejected: {student.rejects}\n')
    return total_welfare


def merge_students(pref_df, grades_df, util_df, pairs_df: pd.DataFrame):
    """Will choose the leader of the pair by the average of the grades"""
    students_dict = create_students(pref_df, grades_df, util_df)
    Student.free_students = set()
    for st_1, st_2 in pairs_df.values:
        if pd.isna(st_2):  # In case st_1 doesn't have a pair
            Student.free_students.add(st_1)
            continue
        avg_grade_1 = np.mean([students_dict[st_1].math_grade, students_dict[st_1].cs_grade])
        avg_grade_2 = np.mean([students_dict[st_2].math_grade, students_dict[st_2].cs_grade])
        if avg_grade_1 > avg_grade_2:
            students_dict[st_1].pair = students_dict[st_2]
            students_dict[st_2].main_partner = False
            Student.free_students.add(st_1)
        else:
            students_dict[st_2].pair = students_dict[st_1]
            students_dict[st_1].main_partner = False
            Student.free_students.add(st_2)
    return students_dict


def task_two(pref_df, grades_df, utils_df, pairs_df):
    students_dict = merge_students(pref_df, grades_df, utils_df, pairs_df)
    projects_dict = create_projects(pref_df)
    deferred_acceptance(students_dict, projects_dict)
    total_welfare = 0
    for sid, student in students_dict.items():
        if student.is_free():
            print('wtf!')
            sys.exit(sid)
        total_welfare += student.get_utility()
        # print(f'Student {sid} -> {student.project.pid}')
        # print(f'utility: {student.get_utility() :.2f}')
        # print(f'rejected: {student.rejects}\n')
    return total_welfare


def market_clearing():
    # TODO: use utils here to construct market clearing prices
    x = 1


def main(n):
    print(f'running for dataset {n}')
    grades_df = pd.read_csv(f'data/train_data/grades_{n}.dat', index_col='student_id')
    pref_df = pd.read_csv(f'data/train_data/students_pref_{n}.dat', index_col='student_id')
    utils_df = pd.read_csv(f'data/train_data/util_{n}.dat', index_col='student_id')
    pairs_df = pd.read_csv(f'data/train_data/pairs_{n}.dat', index_col=False, na_filter=True)
    welfare_1 = task_one(pref_df, grades_df, utils_df)
    welfare_2 = task_two(pref_df, grades_df, utils_df, pairs_df)
    print(welfare_1)
    print(welfare_2)


if __name__ == '__main__':
    for i in range(3, 9):
        main(i)
    # main(5)
