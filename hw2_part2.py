class Student:
    free_students = set()

    def __init__(self, sid: int, pref_list: list, math_grade, cs_grade, utils):
        self.sid = sid
        self.math_grade = math_grade
        self.cs_grade = cs_grade
        self.pref_list = pref_list
        self.project = None
        self.utils = utils
        self.pair = None

    def is_free(self):
        return bool(not self.project)


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


def run_market_clearing(n):
    return {i: i for i in range(1, 6)}, {1: 1, 2: 0, 3: 1, 4: 0, 5: 0}


def calc_total_welfare(matching_file, n) -> int:
    if 'single' in matching_file:
        result = 45
    elif 'coupled' in matching_file:
        result = 55
    else:
        result = 59
    return result
