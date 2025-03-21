import sys
from collections import defaultdict

import numpy as np
import pandas as pd


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
        self.values = utils

    def assign_proj(self, project):
        self.project = project
        Student.free_students.discard(self.sid)
        if self.pair:
            self.pair.project = project

    def is_free(self):
        return bool(not self.project)

    def get_top_pref(self):
        return self.top_pref

    def set_top_pref(self):
        """Assuming {} can only appear at the end of the list of preferences"""
        self.top_pref = self.pref_list[self.rejects]

    def update_values_by_prices(self, projects_dict):
        for i, pid in enumerate(self.pref_list[:-1]):
            self.values[i] = self.utils[i] - projects_dict[pid].price

    def get_top_projects_by_util(self) -> set:
        indices = set()
        top_pids = set()
        max_val = max(self.utils)
        for j, val in enumerate(self.utils):
            if val == max_val:
                indices.add(j)
        for j in indices:
            top_pids.add(self.pref_list[j])
        return top_pids

    def reject_student(self):
        self.project = None
        Student.free_students.add(self.sid)
        if self.rejects < self.max_rejects:
            self.rejects += 1
        self.set_top_pref()

    def test_offer(self, proj_nominee: int):
        """Returns True if the input project is preferable"""
        try:
            candidate_pref = self.pref_list.index(proj_nominee)
        except ValueError:
            return False
        try:
            cur_pref = self.pref_list.index(self.project.pid)
        except ValueError:
            cur_pref = self.max_rejects
        return True if cur_pref > candidate_pref else False

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
        self.main_student = None
        self.partner_student = None
        self.price = 0

    def is_free(self):
        return bool(not self.main_student)

    def accept_offer(self, student):
        if self.main_student:
            self.main_student.reject_student()
        self.main_student = student
        student.assign_proj(self)
        if student.pair:
            self.partner_student = student.pair

    def test_offer(self, nominee, student_candidate=None):
        student = student_candidate if student_candidate else self.main_student
        if getattr(nominee, self.grade_type) > getattr(student, self.grade_type):
            return True
        else:
            return False

    def add_offer(self, student):
        self.proposals[student.sid] = getattr(student, self.grade_type)

    def raise_price(self):
        self.price += 1


def create_students_old(df, grades_df, utils_df):
    students_dict = {}
    for sid, pref_sr in df.iterrows():
        students_dict[sid] = Student(sid, list(pref_sr), grades_df.loc[sid].math_grades, grades_df.loc[sid].cs_grades,
                                     utils_df.loc[sid])
        Student.free_students.add(sid)
    return students_dict


def create_students(df, grades_df):
    students_dict = {}
    for sid, pref_sr in df.iterrows():
        desc_sr = pref_sr.sort_values(ascending=False)
        students_dict[sid] = Student(sid, desc_sr.index.tolist(), grades_df.loc[sid].math_grades,
                                     grades_df.loc[sid].cs_grades, desc_sr.tolist())
        Student.free_students.add(sid)
    return students_dict


def create_projects(df):
    projects_dict = {}
    for project_id in df.columns:
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
        if project.is_free():
            project.accept_offer(students_dict[top_sid])
        elif project.test_offer(students_dict[top_sid]):
            project.accept_offer(students_dict[top_sid])
        else:
            students_dict[top_sid].reject_student()


def deferred_acceptance(students_dict, projects_dict):
    while Student.free_students:
        make_offers(students_dict, projects_dict)
        respond_offers(students_dict, projects_dict)


def write_matching(matching: dict, task):
    df = pd.DataFrame.from_dict(matching, orient='index')
    df.to_csv(f'matching_task_{task}.csv', header=['pid'], index_label='sid')
    print(f'\nmatching_task_{task}.csv was written\n')


def task_one(pref_df, grades_df):
    students_dict = create_students(pref_df, grades_df)
    projects_dict = create_projects(pref_df)
    deferred_acceptance(students_dict, projects_dict)
    total_welfare = 0
    matching = {}
    for sid, student in students_dict.items():
        if student.is_free():
            print('wtf!')
            sys.exit(sid)
        matching[sid] = student.project.pid
        total_welfare += student.get_utility()
    find_blocking_pairs(students_dict)
    write_matching(matching, 1)
    return total_welfare


def merge_students(pref_df, grades_df, pairs_df: pd.DataFrame):
    """Will choose the leader of the pair by the average of the grades"""
    students_dict = create_students(pref_df, grades_df)
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


def task_two(pref_df, grades_df, pairs_df):
    students_dict = merge_students(pref_df, grades_df, pairs_df)
    projects_dict = create_projects(pref_df)
    deferred_acceptance(students_dict, projects_dict)
    total_welfare = 0
    matching = {}
    for sid, student in students_dict.items():
        if student.is_free():
            print('wtf!')
            sys.exit(sid)
        matching[sid] = student.project.pid
        total_welfare += student.get_utility()
    find_blocking_pairs(students_dict)
    write_matching(matching, 2)
    return total_welfare


def test_blocking(student, suspected_blocking, students_dict):
    """If found blocking pair, both students are removed from the suspects set and return True"""
    cur_project = student.project
    counter = 0
    for suspect_id in suspected_blocking:
        suspect = students_dict[suspect_id]
        if suspect is student:
            continue
        if cur_project.test_offer(suspect, student) and student.test_offer(suspect.project.pid):
            if suspect.test_offer(cur_project.pid) and suspect.project.test_offer(student, suspect):
                # print('Blocking!')
                # print(f'sid1: {student.sid} sid2: {suspect_id}')
                # print(f'pid1: {student.project.pid} pid2: {suspect.project.pid}\n')
                counter += 1
    return counter


def find_blocking_pairs(students_dict):
    suspects = set()
    # check if the main_student got his top priority, if not, might be blocking
    for sid, student in students_dict.items():
        if student.pref_list[0] == student.project.pid:
            continue
        suspects.add(sid)
    counter = 0
    suspects_copy = suspects.copy()
    for sid in suspects_copy:
        counter += test_blocking(students_dict[sid], suspects, students_dict)
        suspects.remove(sid)

    print(f'number of blocking pairs {counter}')


def update_students_values(students_dict, projects_dict):
    for sid, student in students_dict.items():
        student.update_values_by_prices(projects_dict)


def make_edges(students_dict):
    edges_st_pr = defaultdict(set)
    edges_pr_st = defaultdict(set)
    for sid, student in students_dict.items():
        top_projs = student.get_top_projects_by_util()
        edges_st_pr[sid].update(top_projs)
        for pid in top_projs:
            edges_pr_st[pid].add(sid)
    return edges_st_pr, edges_pr_st


def market_clearing(pref_df, grades_df, util_df):
    students_dict = create_students(pref_df, grades_df, util_df)
    projects_dict = create_projects(pref_df)

    update_students_values(students_dict, projects_dict)
    edges_st_pr, edges_pr_st = make_edges(students_dict)
    constricted_set = set()
    suspects = set()
    for pid, st_set in edges_pr_st.items():
        if len(st_set) > 1:
            suspects.add(pid)
            neighb = len(st_set)
            for sid in st_set:
                suspects.update(pid)
    print(suspects)
    print(edges_pr_st)
    # TODO: need to implement BFS to look for matching and constricted sets
    # print(len(edges_pr_st))
    # print(edges_st_pr)
    # print(len(edges_st_pr))


def main(n):
    print(f'running for dataset {n}')
    grades_df = pd.read_csv(f'data/train_data/grades_{n}.csv', index_col='student_id')
    # pref_df = pd.read_csv(f'data/train_data/students_pref_{n}.csv', index_col='student_id')
    # utils_df = pd.read_csv(f'data/train_data/util_{n}.csv', index_col='student_id')
    pairs_df = pd.read_csv(f'data/train_data/pairs_{n}.csv', index_col=False, na_filter=True)
    new_df = pd.read_csv(f'data/train_data/preferences_{n}.csv', index_col='student_id')
    new_df.columns = new_df.columns.astype(int)
    welfare_1 = task_one(new_df, grades_df)
    welfare_2 = task_two(new_df, grades_df, pairs_df)
    print(welfare_1)
    print(welfare_2)
    # market_clearing(pref_df, grades_df, utils_df)


if __name__ == '__main__':
    for i in range(1, 7):
        main(i)
    main('test')
    # main(5)
