# merge pickle files into one dict

import bisect
import pickle
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from functools import partial
from math import sqrt
import math
import numpy.random as npr
from typing import NamedTuple


# load pickle files
from scipy.interpolate import InterpolatedUnivariateSpline


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_all_pickle():
    files = os.listdir(os.getcwd())
    # filter out non-pickle files
    files = [file for file in files if file.endswith(".pickle")]
    # load all pickle files,merge dicts
    dict = {}
    for file in files:
        dict.update(load_pickle(file))
    
    # rewrite list as [((lower,upper),[(sigma,fee),(sigma,fee)] ]
    res = {}
    for key, value in dict.items():
        if (key[0], key[1]) in res:
            res[(key[0], key[1])].append((key[2], value))
        else:
            res[(key[0], key[1])] = [(key[2], value)]
    
    # sort the list by lower bound, then upper bound, then sigma
    sorted_res = sorted(res.items(), key=lambda x: (x[0][0], x[0][1], x[1][0][0]))
    return sorted_res


def find_nearest_combinations_index(lower, upper, lower_list, upper_list):
    # Find the index of lower and upper in their respective lists
    lower_index = bisect.bisect_left(lower_list, lower)
    upper_index = bisect.bisect_left(upper_list, upper)

    # Find the nearest points
    lower_indices = [max(0, lower_index-1), min(len(lower_list)-1, lower_index+1)]
    upper_indices = [max(0, upper_index-1), min(len(upper_list)-1, upper_index+1)]

    # Generate the combinations
    combinations = [(l, u) for l in lower_indices for u in upper_indices]

    return combinations


def get_iv_by_interpolation(fee, sigma_fee_list):
    # sigma_fee_list is a list of (sigma, fee) tuples
    # fee is the fee we want to interpolate
    # return the interpolated implied volatility
    # sort first
    sigma_fee_list = sorted(sigma_fee_list, key=lambda x: x[1])
    # Separate the sigma and fee values
    sigma_values, fee_values = zip(*sigma_fee_list)

    # Create a spline interpolation function
    spline = InterpolatedUnivariateSpline(fee_values, sigma_values)

    # Evaluate the spline at the given fee
    interpolated_sigma = spline(fee)

    return interpolated_sigma


def get_implied_fee_by_interpolation(sigma,sigma_fee_list):
    # sigma_fee_list is a list of (sigma, fee) tuples
    # fee is the fee we want to interpolate
    # return the interpolated implied volatility
    # sort first
    sigma_fee_list = sorted(sigma_fee_list, key=lambda x: x[0])
    # Separate the sigma and fee values
    sigma_values, fee_values = zip(*sigma_fee_list)

    # Create a spline interpolation function
    spline = InterpolatedUnivariateSpline(sigma_values, fee_values)

    # Evaluate the spline at the given fee
    interpolated_fee = spline(sigma)

    return interpolated_fee


def get_iv(lower, upper, fee, iv_table ):
    lower_list = sorted(set([x[0][0] for x in iv_table]))
    upper_list = sorted(set([x[0][1] for x in iv_table]))
    # Find the nearest combinations
    combinations_index = find_nearest_combinations_index(lower, upper, lower_list, upper_list)
    # get list in res
    #values = [iv_table[lower_index*len(lower_list)+upper_index][0] for (lower_index,upper_index) in combinations_index]
    combinations_list = [iv_table[lower_index*len(lower_list)+upper_index][1] for (lower_index,upper_index) in combinations_index]

    ivs = [get_iv_by_interpolation(fee,sigma_fee_list) for sigma_fee_list in combinations_list]

    return np.mean(ivs)

def get_implied_fee(lower,upper,sigma,iv_table):
    lower_list = sorted(set([x[0][0] for x in iv_table]))
    upper_list = sorted(set([x[0][1] for x in iv_table]))

    combinations_index = find_nearest_combinations_index(lower, upper, lower_list, upper_list)
    # get list in res
    #values = [iv_table[lower_index*len(lower_list)+upper_index][0] for (lower_index,upper_index) in combinations_index]
    combinations_list = [iv_table[lower_index*len(lower_list)+upper_index][1] \
                         for (lower_index,upper_index) in combinations_index]
    fees = [get_implied_fee_by_interpolation(sigma,sigma_fee_list) for sigma_fee_list in combinations_list]
    return np.mean(fees)




if __name__ == "__main__":
    # Test

    iv_Table = load_all_pickle()
    sigma_list = sorted(set([x[0] for x in iv_Table[0][1]]))

    lower = 0.7
    upper = 1.2
    fee = 0.000644591850716128
    result = get_iv(lower, upper, fee,iv_Table)
    print(result)


    sigma = 0.49
    result = get_implied_fee(lower,upper,sigma,iv_Table)
    print("impiled fee", result)
