from enum import Enum

import numpy as np

import mip

import networkx as nx

from scipy import optimize

import itertools

import mip

from plotly.subplots import make_subplots
from plotly import graph_objects as go
import plotly.figure_factory as ff

# fun = pointer to fun e.g. delta_hat_fun_1D
# params = defining parameters of function, i.e. d for WCPT
# sample_basis = what to sample against in last dimension
# dim_sample = Dimension of sample in tuple, e.g. (N,M,num_steps)
def sample_generic_fun(fun, params, sample_basis, dim_sample):
    basis_len = len(sample_basis)
    ones_basis = np.ones(basis_len, dtype=np.int)
    params_list = list(itertools.repeat(params,basis_len))
    ret = np.zeros(dim_sample)

    if isinstance(dim_sample, int):
        ret = np.array(list(map(fun, params_list, sample_basis)))
    elif len(dim_sample) == 2:
        for i in range(dim_sample[0]):
            ret[i, :] = list(map(fun, params_list, sample_basis, i * ones_basis))
    elif len(dim_sample) == 3:
        for i in range(dim_sample[0]):
            for j in range(dim_sample[1]):
                ret[i, j, :] = list(map(fun, params_list, sample_basis, i * ones_basis, j * ones_basis))
    return ret

# affine approximation for uniform tasks
# h is affine coefficients
# t time
def delta_bar_fun_ut(h, t):
    return h[0]*t + h[1]

# affine approximation for non uniform tasks
# h is a two dimensional array, rows are each task, columns are the coefficients
# i is the index of the task
def delta_bar_fun_nut(h, t, i):
    return h[i, 0]*t + h[i, 1]

# affine approximation for non uniform tasks and nonuniform machines resulting from method 1
# h is a two dimensional array, rows are each task, columns are the coefficients
# i is the index of the task
# j is the index of the machine
def delta_bar_fun_num_0(h, t, i, j):
    return h[i, 0]*t + h[i, 1]*j + h[i, 2]

# affine approximation for non uniform tasks and nonuniform machines resulting from method 1
# h is a dict with 'h_1' being a vector indexed by task, and 'h_2' is a 2D array, rows are tasks, columns are machines
# i is the index of the task
# j is the index of the machine
def delta_bar_fun_num_1(h, t, i, j):
    return h['h_1'][i]*t + h['h_2'][i, j]

# affine approximation for non uniform tasks and nonuniform machines resulting from method 1
# h is a three dimensional array, first dimension is tasks (size N), second is machines (size M), third is coefficients (size 2)
# i is the index of the task
# j is the index of the machine
def delta_bar_fun_num_2(h, t, i, j):
    return h[i, j, 0]*t + h[i, j, 1]

# WCPT approximation for 1D durations, i.e. uniform tasks
# d is the duration
# t is the time
def delta_hat_fun_1D(d, t):
    return t + d

# WCPT approximation for 1D durations, i.e. either tasks or manchines are nonuniform, but not both
# d is the duration
# t is the time
def delta_hat_fun_2D(d, t, i):
    return t + d[i]

# WCPT approximation for 1D durations, i.e. nonuniform tasks and nonuniform machines
# d is a two dimensional array, rows are tasks, columns are machines
# t is the time
def delta_hat_fun_3D(d, t, i, k):
    return t + d[i, k]
