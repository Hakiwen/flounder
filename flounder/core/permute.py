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

# TODO: Insert this into the permute_P function
def permute_B(B, permutation):
    M = B.shape[0]
    new_B = np.zeros(B.shape)
    for i in range(M):
        for j in range(M):
            new_B[i, j] = B[permutation[i], permutation[j]]
    return new_B

def permute_sample(sample, permutation):
    new_sample = np.zeros(sample.shape)
    for i in range(sample.shape[1]):
        new_sample[:, i, :] = sample[:, permutation[i], :]
    return new_sample

# Permute B with a subset of machines, representing a particular type
def permute_typed_B(B, permutation, type_machines):
    num_type_machines = len(type_machines)
    new_B = B.copy()
    for i in range(num_type_machines):
        for j in range(num_type_machines):
            new_B[type_machines[i], type_machines[j]] = B[permutation[i], permutation[j]]
    return new_B

# Permute a sample with a subset of machines, representing a particular type
def permute_typed_sample(sample, permutation, type_machines):
    # Copy because it is only a partial permutation
    new_sample = sample.copy()
    num_type_machines = len(type_machines)
    for i in range(num_type_machines):
        new_sample[:, type_machines[i], :] = sample[:, permutation[i], :]
    return new_sample
