# -*- coding: utf-8 -*-
import os
import sys
from enum import Enum

import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class AutoStrEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


def default(t, val):
    return val if t is None else t


class IStateDictSaveLoad:
    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def save(self, file_path):
        joblib.dump(self.state_dict(), file_path)

    @classmethod
    def load(cls, file_path, *args, **kwargs):
        db = cls(*args, **kwargs)
        db.load_state_dict(joblib.load(file_path))
        return db
