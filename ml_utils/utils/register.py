# -*- coding: utf-8 -*-
import os
import sys
from functools import wraps
from typing import Sequence

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class Register:
    register_dict = {}

    def __init__(self, group_name, func_name=None):
        self.group_name = group_name
        self.func_name = func_name

    def __call__(self, func):
        @wraps(func)
        def register(f):
            if self.group_name not in self.register_dict.keys():
                self.register_dict[self.group_name] = {}
            if self.func_name is None:
                self.func_name = f.__name__
            assert self.func_name not in self.register_dict[self.group_name].keys()
            self.register_dict[self.group_name][self.func_name] = f
            return f

        return register(func)

    def __class_getitem__(cls, item):
        # print(item)
        if isinstance(item, Sequence):
            if len(item) == 1:
                return cls.register_dict[item]
            if len(item) == 2:
                return cls.register_dict[item[0]][item[1]]
        raise IndexError
