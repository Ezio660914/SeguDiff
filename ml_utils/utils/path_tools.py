# -*- coding: utf-8 -*-
import os
import pickle
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
import inspect

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_stem(path_str):
    """
    same effect to pathlib.Path.stem
    """
    return os.path.splitext(os.path.basename(path_str))[0]


def try_save_pkl(obj, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
            print(f"{file_path} saved")
        return file_path
    except PermissionError:
        tmp_dir = tempfile.gettempdir()
        save_dir = os.path.join(tmp_dir, os.path.basename(os.path.dirname(file_path)))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, os.path.basename(file_path))
        warnings.warn(f"permission denied, try to save file in temp dir: {save_path}")
        try:
            with open(save_path, 'wb') as file:
                pickle.dump(obj, file)
                print(f"{save_path} saved")
            return save_path
        except:
            raise


def copy_and_replace(src, dst, allow_same=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    if os.path.exists(dst):
        if os.path.samefile(src, dst) and allow_same:
            print(f"{dst} already exists, skip copying")
            return
        else:
            os.remove(dst)
    shutil.copy(src, dst)


def is_empty(path):
    if os.path.exists(path):
        # Checking if the directory is empty or not
        return not os.path.isfile(path) and not os.listdir(path)
    else:
        return True


def get_project_dir(project_name="ml_utils", file=None):
    if file is None:
        frame_info = inspect.getframeinfo(inspect.currentframe().f_back)
        file = frame_info.filename
    file_path = Path(os.path.abspath(file))
    while file_path.name != project_name:
        file_path = file_path.parent
        if file_path == file_path.parent:
            break
    return file_path
