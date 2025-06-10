# -*- coding: utf-8 -*-
import multiprocessing
import os
import sys
from typing import Literal

import multiprocess
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def async_run(func, args_array, args_parser=None, desc=None, processes=min(os.cpu_count(), 32), backend: Literal['joblib', 'multiprocess', 'multiprocessing'] = 'joblib'):
    """
    显示进度条的进程并行执行

    :param func: 要执行的函数
    :param args_array: 对于这个列表中的每一个元素，都会作为参数传入func中调用
    :param args_parser: args在传入func前，使用这个函数对args进行处理
    :param desc: 要在进度条显示的信息
    :param processes: 要并行的进程数
    :param backend: 使用joblib还是multiprocessing
    :return:
    """
    if backend == 'joblib':
        p = Parallel(n_jobs=processes, return_as="generator", backend='loky')
        func_list = []
        for args in args_array:
            f = delayed(func)
            if callable(args_parser):
                args = args_parser(args)
            func_list.append(f(*args))
        r = tqdm(p(func_list), desc=desc, total=len(func_list))
        return list(r)
    elif backend == 'multiprocessing':
        pool = multiprocessing.Pool(processes)
        result = [pool.apply_async(func, args_parser(args) if callable(args_parser) else args) for args in args_array]
        pool.close()
        result = [r.get() for r in tqdm(result, desc)]
        pool.join()
        return result
    elif backend == 'multiprocess':
        pool = multiprocess.Pool(processes)
        result = [pool.apply_async(func, args_parser(args) if callable(args_parser) else args) for args in args_array]
        pool.close()
        result = [r.get() for r in tqdm(result, desc)]
        pool.join()
        return result
    else:
        raise ValueError('Unknown backend: {}'.format(backend))
