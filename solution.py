import pandas as pd


class Student:
    def __init__(self, sid: int, pref_list: list):
        self.sid = sid
        self.math_grade = None
        self.cs_grade = None
        self.pref_list = pref_list
        self.max_rejects = len(pref_list)
        self.project = None
        self.top_pref = None
        self.rejects = 0

    def assign_proj(self, proj_id):
        self.project = proj_id

    def is_free(self):
        return bool(not self.project)

    def get_top_pref(self):
        return self.top_pref

    def set_top_pref(self):
        if self.rejects < self.max_rejects:
            temp = self.pref_list[self.rejects]
            if type(temp) is int:
                self.top_pref = temp
            elif type(temp) is set:
                self.top_pref = temp.pop()
        else:
            self.top_pref = self.pref_list #TODO: continue here

    def reject_student(self):
        self.project = None
        if self.rejects < self.max_rejects:
            self.rejects += 1
        self.set_top_pref()


if __name__ == '__main__':
    students_dict = {}
    df = pd.read_csv('data/train_data/students_pref_3.dat', index_col='student_id')
    for sid, pref_sr in df.iterrows():
        students_dict[sid] = Student(sid, list(pref_sr))
