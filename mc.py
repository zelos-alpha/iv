import math
import pickle
from math import sqrt
import random
import itertools
import sys,os

# os.chdir(os.getcwd()+"/uniswap_implied_vol")
from typing import NamedTuple

import numpy as np
import numpy.random as npr
from functools import partial
import pandas as pd
from multiprocessing import Pool
import time
from tqdm import tqdm  # 导入tqdm库
# steps:
# 1. generate pathes
# 2. calculated implied fees
# 3. make it as a big dict (key: (lower,upper,sigma), value: implied fees)
# 4. iter by sigma


npr.seed(0)
np.set_printoptions(precision=5)
days = 7
step_length_per_day = 1440


class LP(NamedTuple):
    lower_rate:float
    upper_rate:float

def calcul_L(lower,upper):
    return 1/(2-1/sqrt(upper)-sqrt(lower))

def generate_pathes(sigma, I=10000, dt=1 / 24 / 60 / 365, T=days / 365):
    M = int(T / dt)
    S = np.zeros((M + 1, I))
    S[0] = 1
    for t in range(1, M + 1):
        S[t] = S[t - 1] * np.exp((- 0.5 * sigma ** 2) * dt +
                                 sigma * math.sqrt(dt) * npr.standard_normal(I))
    return S


def get_lp_value(one_lp, st):
    virtual_l = calcul_L(one_lp.lower_rate, one_lp.upper_rate)
    _lower = one_lp.lower_rate
    _upper = one_lp.upper_rate

    def lp_vlaue(st):
        if st < _lower:
            return virtual_l * st * (1 / sqrt(_lower) - 1 / sqrt(_upper))
        elif st > _upper:
            return virtual_l * (sqrt(_upper) - sqrt(_lower))
        else:
            return virtual_l * (2 * sqrt(st) - sqrt(_lower) - st / sqrt(_upper))

    return lp_vlaue(st)


def get_life_span(path, one_lp):
    upper_index = np.argmax(path > one_lp.upper_rate)
    lower_index = np.argmax(path < one_lp.lower_rate)
    if upper_index == lower_index == 0:
        life_span = len(path) - 1
    else:
        if upper_index == 0:
            upper_index = np.inf
        elif lower_index == 0:
            lower_index = np.inf

        life_span = min(upper_index, lower_index)
    return life_span


def get_average_step_fees(lower_rate,upper_rate, pathes):
    one_lp = LP(lower_rate,upper_rate)
    this_lp_vlaue = partial(get_lp_value, one_lp)

    def get_implied_step_fee(one_path):
        life_span = get_life_span(one_path, one_lp)
        if life_span != 0:
            return (1 - this_lp_vlaue(one_path[life_span])) /life_span
        else:
            return None

    impiled_step_fees = np.apply_along_axis(get_implied_step_fee, axis=0, arr=pathes)
    expect_fee = np.average(impiled_step_fees)*step_length_per_day
    return expect_fee

        

def get_square_range_neg(start, end, num, power):
    """
    把数字序列, 从等差序列变化为x^2(-1~0)区间的分布
    比如: 输入0.5 ~ 1, 间隔0.1, 结果为0.5,0.68,0.82,0.92,0.98,1
    目的是使得结果序列在start这边稀疏一些, end这边密集一些
    """
    amp = (end - start)
    interval = amp / (num - 1)
    return np.flip(1 - (np.arange(0, 1.00001, interval / amp) ** power)) * amp + start


def get_square_range(start, end, num, power):

    amp = (end - start)
    interval = amp / (num - 1)
    return (np.arange(0, 1.00001, interval / amp) ** power) * amp + start

def calculate_vol(vol,iter_list):
    vol_str = "{:.2f}".format(round(vol,2))
    save_path = f"sigma_{vol_str}.pickle"
    pathes = generate_pathes(vol)
    big_dict = {}
    for lower_rate, upper_rate in iter_list:
        expect_fee = get_average_step_fees(lower_rate, upper_rate, pathes)
        big_dict[(lower_rate, upper_rate, vol)] = expect_fee
    with open(save_path, "wb") as f:
        pickle.dump(big_dict, f)


if __name__ == "__main__":
    config = {
        "multi_thread": True
    }

    # generate big dict
    # key: (lower,upper,sigma), value: implied fees
    lower_rate_bound = 0.5
    upper_rate_bound = 2
    lower = get_square_range_neg(lower_rate_bound, 0.999, 20, 2)
    upper = get_square_range(1.001, upper_rate_bound, 20, 2)
    iter_list = list(itertools.product(lower, upper))
    vols = np.arange(0.2, 1.25, 0.05)  # year vol rate.
    print(iter_list)
    if config["multi_thread"]:
        with Pool() as p:
            with tqdm(total=len(vols)) as pbar:
                for _ in p.imap_unordered(partial(calculate_vol, iter_list=iter_list), vols):
                    pbar.update(1)