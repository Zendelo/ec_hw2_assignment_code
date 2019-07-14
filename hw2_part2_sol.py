import sys

import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

data_dir = 'data/train_data_comp/'


# data_dir = 'data/train_data/'
# data_dir = 'data/'


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
        self.values = utils.copy()

    def assign_proj(self, project):
        self.project = project
        Student.free_students.discard(self.sid)
        if self.pair:
            self.pair.project = project

    def is_free(self):
        return bool(not self.project)

    def get_top_pref(self):
        m = max(self.values)
        max_indices = [i for i, j in enumerate(self.values) if j == m]
        if len(max_indices) > 1:
            top_pref = set()
            for i in max_indices:
                top_pref.add(self.pref_list[i])
        else:
            top_pref = self.pref_list[max_indices[0]]
        return top_pref

    def set_top_pref(self):
        self.top_pref = self.pref_list[self.rejects]

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
        try:
            idx = self.pref_list.index(self.project.pid)
        except ValueError:
            idx = self.max_rejects
        return self.utils[idx]

    def update_values_by_prices(self, projects_dict):
        for i, pid in enumerate(self.pref_list):
            self.values[i] = self.utils[i] - projects_dict[pid].price


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


def task_one(pref_df, grades_df):
    students_dict = create_students(pref_df, grades_df)
    projects_dict = create_projects(pref_df)
    deferred_acceptance(students_dict, projects_dict)
    matching = {}
    for sid, student in students_dict.items():
        if student.is_free():
            print('Error! missing matching')
            sys.exit(sid)
        matching[sid] = student.project.pid
    return matching


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


def test_blocking(student, suspected_blocking, students_dict):
    cur_project = student.project
    counter = 0
    for suspect_id in suspected_blocking:
        suspect = students_dict[suspect_id]
        if suspect is student:
            continue
        if cur_project.test_offer(suspect, student) and student.test_offer(suspect.project.pid):
            if suspect.test_offer(cur_project.pid) and suspect.project.test_offer(student, suspect):
                counter += 1
    return counter


def find_blocking_pairs(students_dict):
    suspects = set()
    # check if the student got his top priority, if not, might be blocking
    for sid, student in students_dict.items():
        if student.pref_list[0] == student.project.pid:
            continue
        suspects.add(sid)
    counter = 0
    suspects_copy = suspects.copy()
    for sid in suspects_copy:
        counter += test_blocking(students_dict[sid], suspects, students_dict)
        suspects.remove(sid)
    return counter


def update_students_values(students_dict, projects_dict):
    for student in students_dict.values():
        student.update_values_by_prices(projects_dict)


def make_edges(students_dict):
    edges = {}
    for sid, student in students_dict.items():
        pid = student.get_top_pref()
        if type(pid) is int:
            edges[sid] = [str(pid)]
        elif type(pid) is set:
            edges[sid] = [str(p) for p in pid]
    return edges


def is_constricted(susp_set, graph):  # TODO: Look for edge cases, something here doesn't work
    students = set([i for li in susp_set.values() for i in li])
    for i in range(2, len(students) + 1):
        sub_sets = combinations(students, i)
        for sub_set in sub_sets:
            neighbors = set()
            for sid in sub_set:
                neighbors.update(set(graph.neighbors(sid)))
            if len(sub_set) > len(neighbors):
                return True
    return False


def find_constricted_set(graph, projects):
    const_set = set()
    suspect_dict = {}
    for pid in projects:
        if graph.degree(pid) > 1:
            suspect_dict[pid] = list(graph.neighbors(pid))
    # if is_constricted(suspect_dict, graph):
    #     const_set = suspect_dict.keys()
    #     print('all set')
    # else:
    #     print('NOT all set')
    #     for i in range(1, len(suspect_dict)):
    #         sub_sets = combinations(suspect_dict, i)
    #         for sub_set in sub_sets:
    #             if is_constricted({k: suspect_dict[k] for k in sub_set}, graph):
    #                 return sub_set
    for i in range(1, len(suspect_dict) + 1):
        sub_sets = combinations(suspect_dict, i)
        if i > 3:
            print(i)
        for sub_set in sub_sets:
            if is_constricted({k: suspect_dict[k] for k in sub_set}, graph):
                return sub_set
    return const_set


def update_prices(const_set, projects_dictionary):
    for pid in const_set:
        projects_dictionary[int(pid)].price += 1


def market_clearing(pref_df, grades_df):
    students_dict = create_students(pref_df, grades_df)
    # students_dict = merge_students(pref_df, grades_df, pairs_df)
    projects_dict = create_projects(pref_df)
    while True:
        update_students_values(students_dict, projects_dict)
        edges = make_edges(students_dict)
        graph = nx.Graph(edges)
        students, projects = nx.bipartite.sets(graph, students_dict)
        max_matching = nx.bipartite.maximum_matching(graph, students_dict)
        # max_matching_dict = {k: max_matching[k] for k in students_dict}
        # max_matching = nx.max_weight_matching(graph)
        # max_matching = nx.maximal_matching(graph)
        if len(max_matching) == len(students) * 2:
            break
        constricted_set = find_constricted_set(graph, projects)
        if not constricted_set:
            exit(max_matching)
        # print(f'const set {constricted_set}')
        update_prices(constricted_set, projects_dict)
    # print(max_matching)
    return {k: max_matching[k] for k in students_dict}, {pid: project.price for pid, project in projects_dict.items()}


def run_market_clearing(n):
    # pairs_df = pd.read_csv(f'{data_dir}/pairs_{n}.csv', index_col=False, na_filter=True)
    grades_df = pd.read_csv(f'{data_dir}/grades_{n}.csv', index_col='student_id')
    preferences_df = pd.read_csv(f'{data_dir}/preferences_{n}.csv', index_col='student_id')
    preferences_df.columns = preferences_df.columns.astype(int)
    matching_dict, prices_dict = market_clearing(preferences_df, grades_df)
    # return {1: 2, 2: 3, 3: 4, 4: 1, 5: 5}
    # print(matching_dict)
    return matching_dict, prices_dict


def reconstruct_matching(students_dict, projects_dict, matching_df: pd.DataFrame):
    for sid, pid in matching_df.itertuples():
        student = students_dict[sid]
        project = projects_dict[pid]
        project.accept_offer(student)


def reconstruct_matching_pairs(students_dict, projects_dict, matching_df):
    for sid, pid in matching_df.itertuples():
        student = students_dict[sid]
        project = projects_dict[pid]
        if student.main_partner:
            project.accept_offer(student)
        else:
            project.partner_student = student
            student.assign_proj(project)


def recon_matching(matching_file, n):
    grades_df = pd.read_csv(f'{data_dir}/grades_{n}.csv', index_col='student_id')
    pairs_df = pd.read_csv(f'{data_dir}/pairs_{n}.csv', index_col=False, na_filter=True)
    preferences_df = pd.read_csv(f'{data_dir}/preferences_{n}.csv', index_col='student_id')
    preferences_df.columns = preferences_df.columns.astype(int)
    matching_df = pd.read_csv(matching_file, index_col='sid').sort_index()
    projects_dict = create_projects(preferences_df)
    if 'coupled' in matching_file:
        students_dict = merge_students(preferences_df, grades_df, pairs_df)
        reconstruct_matching_pairs(students_dict, projects_dict, matching_df)
    else:
        students_dict = create_students(preferences_df, grades_df)
        reconstruct_matching(students_dict, projects_dict, matching_df)
    return students_dict, projects_dict


def count_blocking_pairs(matching_file, n) -> int:
    students_dict, projects_dict = recon_matching(matching_file, n)
    blocking_pairs = find_blocking_pairs(students_dict)
    return blocking_pairs
    # return 0 if '1' in matching_file else 1


def calc_total_welfare(matching_file, n) -> int:
    students_dict, projects_dict = recon_matching(matching_file, n)
    total_welfare = 0
    for sid, student in students_dict.items():
        if student.is_free():
            print('Error! missing matching')
            sys.exit(sid)
        total_welfare += student.get_utility()
    # return 73 if 'single' in matching_file else 63
    return total_welfare


def main(n):
    run_market_clearing(n)


if __name__ == '__main__':
    # for i in range(1, 5):
    #     main(i)
    main('test')
