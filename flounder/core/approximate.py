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

# X \in R^MxN, returns hyper x \in R^N
# for writing, better definition of a hyperplane
# Also determine best objective function and complexity of this
def upperbounding_hyperplane(A, b):
    A = np.array(A)
    b = np.array(b)
    N = A.shape[1]
    M = A.shape[0]
    assert(b.shape[0] == M)
    # print(N)
    # print(M)
    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    e = [model.add_var(name='e({})'.format(i+1)) for i in range(M)]
    x = [model.add_var(name='x({})'.format(i+1)) for i in range(N)]
    #     e = A*x - b
    for i in range(M):
        e[i] = -1*b[i]
        for j in range(N):
            e[i] = e[i] + A[i, j]*x[j]
        model += e[i] >= 0
    total_error = mip.xsum(e[i] for i in range(M))
    model.objective = total_error
    model.optimize()
    #     x_found = np.zeros(N)
    x_found = np.array([x[i].x for i in range(N)])
    # print(x_found)
    return x_found, model.objective_value

# t is a vector of the time samples, common for all rows of B
# B is the samples, b_ij = the jth sample for the ith machine
# returns (h_1, vector of h_2)
def het_qp(t, B):
    t = np.array(t)
    B = np.array(B)
    n = t.shape[0]
    m = B.shape[0]
    assert(B.shape[1] == n)

    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0
    e = [[model.add_var(name='e({},{})'.format(i+1, j+1)) for j in range(n)]for i in range(m)]
    h_1 = model.add_var(name='h_1')
    h_2 = [model.add_var(name='h_2({})'.format(i + 1)) for i in range(m)]
    # total_error = model.add_var(name='e_sum')
    total_error = 0
    for i in range(m):
        for j in range(n):
            if B[i][j] <= t[-1]:
                e[i][j] = t[j]*h_1 + h_2[i] - B[i][j]
                model += e[i][j] >= 0
                total_error = total_error + e[i][j]

    model.objective = total_error
    model.optimize()
    h1_found = h_1.x
    h2_found = np.array([h_2[i].x for i in range(m)])

    return h1_found, h2_found

def find_approximation_window(delta_sample, t_sample, H):
    ds_index = (delta_sample <= H)
    ds = delta_sample[ds_index]
    ts = t_sample[ds_index]
    return ds, ts

def find_single_WCPT(delta_sample, t_sample, H):
    d = None
    ds, ts = find_approximation_window(delta_sample, t_sample, H)
    if ds.shape[0] > 0:
        d = np.max(ds - ts)
    else:
        d = H + 1
    return d

def find_WCPT(delta_sample, t_sample, H):
    d = None
    if len(delta_sample.shape) == 1:
        d = find_single_WCPT(delta_sample, t_sample, H)
    elif len(delta_sample.shape) == 2:
        d = np.zeros((delta_sample.shape[0]))
        for i in range(delta_sample.shape[0]):
            d[i] = find_single_WCPT(delta_sample[i, :], t_sample, H)
    elif len(delta_sample.shape) == 3:
        d = np.zeros((delta_sample.shape[0], delta_sample.shape[1]))
        for i in range(delta_sample.shape[0]):
            for j in range(delta_sample.shape[1]):
                d[i, j] = find_single_WCPT(delta_sample[i, j, :], t_sample, H)
    return d

# Approximates single dimension of delta
# TODO: evaluate if ts[-1] should be changed to H
def approximate_single_delta(d, delta_sample, H, t_sample):
    ds, ts = find_approximation_window(delta_sample, t_sample, H)
    this_sample = np.append(ds, ts[-1] + d)
    ones = np.ones(this_sample.shape)
    this_time = np.append(ts, ts[-1])
    this_in = np.column_stack((this_time, ones))
    h, total_approx_error = upperbounding_hyperplane(this_in, this_sample)
    return h, total_approx_error

# approximates delta for a uniform task problem, pass through atm
def approximate_delta_ut(d, delta_sample, H, t_sample):
    return approximate_single_delta(d, delta_sample, H, t_sample)

# approximates delta for a nonuniform task
def approximate_delta_nut(d, delta_sample, H, N, t_sample):
    h_dim = 2
    h = np.zeros((N, h_dim))
    total_approx_error = np.zeros(h.shape[:-1])
    for i in range(N):
        h[i], total_approx_error[i] = approximate_single_delta(d[i], delta_sample[i, :], H, t_sample)
    return h, total_approx_error

# approximates delta for a set of nonuniform machines using method 1
def approximate_delta_num_0(d, delta_sample, H, N, num_steps, M, t_sample):
    # We get to choose our specific ordering of p, so we can also optimize over the reorderings of p

    # permutation_list = list(itertools.permutations([i for i in range(M)]))

    # New permutation technique, reducing set of permutations to ordering of WCPT

    permutation_list = []
    for i in range(N):
        permutation_list.append(np.argsort(d[i, :]))

    num_perm = len(permutation_list)
    h_dim = 3
    perm_h = np.zeros((num_perm, N, h_dim))
    # ones = np.ones(M*num_steps)

    # num_steps + 1 for the additional sample that enforces the intercept constraint
    ones = np.ones(M*(num_steps + 1))
    perm_total_approx_error = np.zeros((num_perm, N))
    for permdex in range(num_perm):
        for i in range(N):
            this_sample = []
            this_p_index = []
            this_t_sample = []
            for k in range(M):
                this_sample = np.concatenate((this_sample, delta_sample[i, k, :]))
                this_p_index = np.concatenate((this_p_index, permutation_list[permdex][k]*np.ones(num_steps)))
                this_t_sample = np.concatenate((this_t_sample, t_sample))


                this_sample = np.append(this_sample, H + d[i, k])
                this_p_index = np.append(this_p_index, permutation_list[permdex][k])
                this_t_sample = np.append(this_t_sample, H)

            this_in = np.column_stack((this_t_sample, this_p_index, ones))
            perm_h[permdex, i, :], perm_total_approx_error[permdex, i] = upperbounding_hyperplane(this_in, this_sample)
    perm_sum_approx_error = np.sum(perm_total_approx_error, axis=1)
    argmin_perm = np.argmin(perm_sum_approx_error)
    h = perm_h[argmin_perm, :, :]
    P_permutation = permutation_list[argmin_perm]
    # permute_P()
    return h, P_permutation
#

def approximate_delta_num_1(delta_sample, N, M, t_sample):
    h_1 = np.zeros(N)
    h_2 = np.zeros((N, M))
    for i in range(N):
        this_B = np.array(delta_sample[i, :, :])
        this_B.reshape((M, t_sample.shape[0]))

        qp_res = het_qp(t_sample, this_B)
        h_1[i] = qp_res[0]
        h_2[i, :] = qp_res[1]
    return h_1, h_2

def approximate_delta_num_2(d, delta_sample, H, N, M, t_sample):
    h = np.zeros((N, M, 2))
    error = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            h[i, j, :], error[i, j] = approximate_single_delta(d[i, j], delta_sample[i, j, :], H, t_sample)
    return h, error

def approximate_delta_num_0_het(d, delta_sample, H, N, num_steps, num_types, M, machine_types, t_sample, task_types):
    h_dim = 3
    h = np.zeros((N, 3))
    P_permutation = []
    for u in range(num_types):
        type_u_machines = np.where(machine_types == u)[0]
        num_u_machines = type_u_machines.shape[0]
        type_u_tasks = np.where(task_types == u)[0]
        num_u_tasks = type_u_tasks.shape[0]

        permutation_list = []
        for i in type_u_tasks:
            permutation_list.append(type_u_machines[np.argsort(d[i, type_u_machines])])

        # permutation_list = list(itertools.permutations(type_u_machines))

        num_perm = len(permutation_list)

        perm_h = np.zeros((num_perm, num_u_tasks, h_dim))
        # Additional sample for each point
        ones = np.ones(num_u_machines * (num_steps + 1))
        perm_total_approx_error = np.zeros((num_perm, N))
        for permdex in range(num_perm):
            for i in range(num_u_tasks):
                this_sample = []
                this_p_index = []
                this_t_sample = []
                for k in range(num_u_machines):
                    this_p = permutation_list[permdex][k]
                    this_sample = np.concatenate((this_sample, delta_sample[type_u_tasks[i], this_p, :]))
                    this_p_index = np.concatenate(
                        (this_p_index, this_p * np.ones(num_steps)))
                    this_t_sample = np.concatenate((this_t_sample, t_sample))

                    this_sample = np.append(this_sample, d[type_u_tasks[i], type_u_machines[k]] + H)
                    this_p_index = np.append(this_p_index, this_p)
                    this_t_sample = np.append(this_t_sample, H)
                this_in = np.column_stack((this_t_sample, this_p_index, ones))
                # print(this_in)
                # print(this_in.shape)
                perm_h[permdex, i, :], perm_total_approx_error[permdex, i] = upperbounding_hyperplane(
                    this_in, this_sample)
        perm_sum_approx_error = np.sum(perm_total_approx_error, axis=1)
        argmin_perm = np.argmin(perm_sum_approx_error)
        # print(permutation_list)
        # print(perm_sum_approx_error)
        # print(perm_h[:, 1, :])
        h[type_u_tasks, :] = perm_h[argmin_perm, :, :]
        P_permutation.append(permutation_list[argmin_perm])
    return h, P_permutation
    # permute_P()

def approximate_delta_num_1_het(delta_sample, N, num_types, M, machine_types, t_sample, task_types):
    h_1 = np.zeros(N)
    h_2 = np.zeros((N, M))
    for u in range(num_types):
        type_u_machines = np.where(machine_types == u)[0]
        num_u_machines = type_u_machines.shape[0]
        type_u_tasks = np.where(task_types == u)[0]
        num_u_tasks = type_u_tasks.shape[0]

        for i in type_u_tasks:
            this_B = np.array(delta_sample[i, type_u_machines, :])
            this_B.reshape((num_u_machines, t_sample.shape[0]))

            qp_res = het_qp(t_sample, this_B)
            h_1[i] = qp_res[0]
            h_2[i, type_u_machines] = qp_res[1]
    return h_1, h_2

# pass through of nonuniform machine for
def approximate_delta_num_2_het(d, delta_sample, H, N, M, t_sample):
    return approximate_delta_num_2(d, delta_sample, H, N, M, t_sample)
