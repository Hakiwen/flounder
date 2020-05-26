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

from .types import *

# replicates the processor assignment sum operation
# x is a mip decision variable
def pasum(x, i, M):
    pasum = 0
    for k in range(M):
        pasum += x[i][k].x * k
    return pasum

# replicates processor assignment when assigning time as well
# g is the mip decision variable for assignment
def ptsum(g, i, M, num_steps):
    ret = 0
    for j in range(M):
       for t in range(num_steps):
                    ret += g[i][j][t].x*j
    return ret

# replicates time assignment summation
def stsum(g, i, M, num_steps, t_step):
    ret = 0
    if M > 1:
        for j in range(M):
            for t in range(num_steps):
               ret += g[i][j][t].x*t*t_step
    else:
        for t in range(num_steps):
            ret += g[i][t].x*t*t_step
    return ret



def compute_WCPT_schedule(scheduling_problem, d):
    if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
        scheduling_problem.U = np.zeros((scheduling_problem.N, scheduling_problem.M))
        for i in range(scheduling_problem.N):
            for j in range(scheduling_problem.M):
                if scheduling_problem.task_types[i] == scheduling_problem.machine_types[j]:
                    scheduling_problem.U[i, j] = 1

    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0

    s = [model.add_var(name='s({})'.format(i + 1)) for i in range(scheduling_problem.N)]
    C = [model.add_var(name='C({})'.format(i + 1)) for i in range(scheduling_problem.N)]

    sigma = [[model.add_var(var_type=mip.BINARY, name='sigma({},{})'.format(i + 1, j + 1)) for j in range(scheduling_problem.N)]
             for i in range(scheduling_problem.N)]
    if scheduling_problem.multimachine:
        p = [model.add_var(var_type=mip.INTEGER, name='p({})'.format(i + 1)) for i in range(scheduling_problem.N)]
        x = [[model.add_var(var_type=mip.BINARY, name='x({},{}'.format(i + 1, k + 1)) for k in range(scheduling_problem.M)]
             for i in range(scheduling_problem.N)]
        epsilon = [
            [model.add_var(var_type=mip.BINARY, name='epsilon({},{})'.format(i + 1, j + 1)) for i in range(scheduling_problem.N)]
            for j in range(scheduling_problem.N)]
        proc_assign_sum = {}

    if scheduling_problem.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
        gamma = [
            [
                [
                    [
                        model.add_var(var_type=mip.BINARY, name='z({},{},{},{})'.format(i + 1, j + 1, h + 1, k + 1))
                        for k in range(scheduling_problem.M)
                    ]
                    for h in range(scheduling_problem.M)
                ]
                for j in range(scheduling_problem.N)
            ]
            for i in range(scheduling_problem.N)
        ]

    for i in range(scheduling_problem.N):
        model += s[i] >= 0
        model += C[i] <= scheduling_problem.W
        if scheduling_problem.multimachine:
            model += p[i] >= 0
            model += p[i] <= scheduling_problem.M - 1
            p[i] = 0
            proc_assign_sum[i] = 0
            for k in range(scheduling_problem.M):
                p[i] = p[i] + k * x[i][k]
                proc_assign_sum[i] = proc_assign_sum[i] + x[i][k]
            model += proc_assign_sum[i] == 1

        if len(scheduling_problem.delta_sample.shape) == 1:
            C[i] = s[i] + scheduling_problem.d
        elif len(scheduling_problem.delta_sample.shape) == 2:
            C[i] = s[i] + scheduling_problem.d[i]
        # TODO:  change to check for specifically het machines vs het tasks, the above is only valid for tasks
        elif len(scheduling_problem.delta_sample.shape) == 3:
            C[i] = s[i]
            for j in range(scheduling_problem.M):
                C[i] = C[i] + scheduling_problem.d[i, j] * x[i][j]

    if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
        for i in range(scheduling_problem.N):
            for j in range(scheduling_problem.M):
                model += x[i][j] <= scheduling_problem.U[i][j]

    z = model.add_var(name='z')
    for i in range(scheduling_problem.N):
        model += z >= C[i]

    model.objective = z

    for i in range(scheduling_problem.N):
        for j in range(scheduling_problem.N):
            if j != i:
                if not scheduling_problem.multimachine:
                    model += sigma[i][j] + sigma[j][i] == 1
                else:
                    model += sigma[i][j] + sigma[j][i] <= 1
                    model += epsilon[i][j] + epsilon[j][i] <= 1
                    model += epsilon[i][j] + epsilon[j][i] + sigma[i][j] + sigma[j][i] >= 1
                    model += p[j] - p[i] - epsilon[i][j] * (scheduling_problem.M + 1) <= 0
                    model += p[j] - p[i] - 1 - (epsilon[i][j] - 1) * (scheduling_problem.M + 1) >= 0

                model += s[j] - C[i] - (sigma[i][j] - 1) * scheduling_problem.W >= 0
                if scheduling_problem.problem_type.task_relation_type == TaskRelationType.PRECEDENCE:
                    model += sigma[i][j] >= scheduling_problem.A[i, j]

                if scheduling_problem.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
                    for h in range(scheduling_problem.M):
                        for k in range(scheduling_problem.M):
                            model += x[i][h] - gamma[i][j][h][k] >= 0
                            model += x[j][k] - gamma[i][j][h][k] >= 0
                            model += x[i][h] + x[j][k] - 1 - gamma[i][j][h][k] <= 0
                            model += scheduling_problem.A[i][j] * gamma[i][j][h][k] <= scheduling_problem.B[h][k]

    status = model.optimize()

    if status == mip.OptimizationStatus.INFEASIBLE:
        return False

    objective = model.objective_value

    # print(scheduling_problem.WCPT_objective)

    schedule = []

    for i in range(scheduling_problem.N):
        if scheduling_problem.multimachine:
            schedule.append((s[i].x, pasum(x, i, scheduling_problem.M)))
        else:
            schedule.append((s[i].x, 0))

    return schedule, objective


def compute_exact_schedule(scheduling_problem):
    p = {}

    if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
        scheduling_problem.U = np.zeros((scheduling_problem.N, scheduling_problem.M))
        for i in range(scheduling_problem.N):
            for j in range(scheduling_problem.M):
                if scheduling_problem.task_types[i] == scheduling_problem.machine_types[j]:
                    scheduling_problem.U[i, j] = 1

    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0

    # Rectify delta funciton
    # TODO: set H < T and then afford irregular spacing
    scheduling_problem.delta_rect = np.zeros(scheduling_problem.delta_sample.shape)
    if scheduling_problem.multimachine:
        for i in range(scheduling_problem.N):
            for j in range(scheduling_problem.M):
                for t in range(scheduling_problem.num_steps):
                    if scheduling_problem.problem_type.machine_load_type == MachineLoadType.UNIFORM:
                        sample = scheduling_problem.delta_sample[i][t]
                        scheduling_problem.delta_rect[i][t] = scheduling_problem.t_step * np.ceil(sample / scheduling_problem.t_step)
                    else:
                        sample = scheduling_problem.delta_sample[i][j][t]
                        scheduling_problem.delta_rect[i][j][t] = scheduling_problem.t_step * np.ceil(sample / scheduling_problem.t_step)
    else:
        for i in range(scheduling_problem.N):
            for t in range(scheduling_problem.num_steps):
                if scheduling_problem.problem_type.task_load_type == TaskLoadType.UNIFORM:
                    sample = scheduling_problem.delta_sample[t]
                else:
                    sample = scheduling_problem.delta_sample[i][t]
                scheduling_problem.delta_rect[i][t] = scheduling_problem.t_step * np.ceil(sample / scheduling_problem.t_step )

    # overall assignment variable

    if scheduling_problem.multimachine:
        g = [[[model.add_var(var_type=mip.BINARY, name='g({},{},{})'.format(i + 1, j + 1, t + 1)) for t in
               range(scheduling_problem.num_steps)]
              for j in range(scheduling_problem.M)] for i in range(scheduling_problem.N)]
    else:
        g = [[model.add_var(var_type=mip.BINARY, name='g({},{})'.format(i + 1, t + 1)) for t in
              range(scheduling_problem.num_steps)]
             for i in range(scheduling_problem.N)]
    # x = sum over time for task i machine j

    s = [model.add_var(name='s({})'.format(i+1)) for i in range(scheduling_problem.N)]
    C = [model.add_var(name='C({})'.format(i+1)) for i in range(scheduling_problem.N)]

    sigma = [[model.add_var(var_type=mip.BINARY, name='sigma({},{})'.format(i + 1, j + 1)) for i in range(scheduling_problem.N)]
             for j in range(scheduling_problem.N)]

    if scheduling_problem.multimachine:
        p = [model.add_var(var_type=mip.INTEGER, name='p({})'.format(i + 1)) for i in range(scheduling_problem.N)]
        x = [[model.add_var(var_type=mip.BINARY, name='x({},{}'.format(i + 1, k + 1)) for k in range(scheduling_problem.M)]
             for i in range(scheduling_problem.N)]
        epsilon = [
            [model.add_var(var_type=mip.BINARY, name='epsilon({},{})'.format(i + 1, j + 1)) for i in range(scheduling_problem.N)]
            for j in range(scheduling_problem.N)]
        proc_assign_sum = {}

        for i in range(scheduling_problem.N):
            for j in range(scheduling_problem.M):
                x_sum = 0
                for t in range(scheduling_problem.num_steps):
                    x_sum = x_sum + g[i][j][t]
                x[i][j] = x_sum

    if scheduling_problem.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
        gamma = [
            [
                [
                    [
                        model.add_var(var_type=mip.BINARY, name='gamma({},{},{},{})'.format(i + 1, j + 1, h + 1, k + 1))
                        for k in range(scheduling_problem.M)
                    ]
                    for h in range(scheduling_problem.M)
                ]
                for j in range(scheduling_problem.N)
            ]
            for i in range(scheduling_problem.N)
        ]

    for i in range(scheduling_problem.N):
        model += s[i] >= 0
        model += C[i] <= scheduling_problem.W
        if scheduling_problem.multimachine:
            model += p[i] >= 0
            model += p[i] <= scheduling_problem.M - 1
            p[i] = 0
            proc_assign_sum[i] = 0
            for k in range(scheduling_problem.M):
                p[i] = p[i] + k*x[i][k]
                proc_assign_sum[i] = proc_assign_sum[i] + x[i][k]
            model += proc_assign_sum[i] == 1


        if scheduling_problem.multimachine:
            completion_sum = 0
            single_assign = 0
            start_sum = 0
            if scheduling_problem.problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                for j in range(scheduling_problem.M):
                    for t in range(scheduling_problem.num_steps):
                        completion_sum = completion_sum + g[i][j][t]*scheduling_problem.delta_rect[i][j][t]
                        single_assign = single_assign + g[i][j][t]
                        start_sum = start_sum + g[i][j][t]*t*scheduling_problem.t_step
            else:
                for j in range(scheduling_problem.M):
                    for t in range(scheduling_problem.num_steps):
                        completion_sum = completion_sum + g[i][j][t] * scheduling_problem.delta_rect[i][t]
                        single_assign = single_assign + g[i][j][t]
                        start_sum = start_sum + g[i][j][t] * t * scheduling_problem.t_step
            C[i] = completion_sum
            s[i] = start_sum
            model += single_assign == 1
        else:
            completion_sum = 0
            single_assign = 0
            start_sum = 0
            for t in range(scheduling_problem.num_steps):
                completion_sum = completion_sum + g[i][t]*scheduling_problem.delta_rect[i][t]
                single_assign = single_assign + g[i][t]
                start_sum = start_sum + g[i][t]*t*scheduling_problem.t_step
            C[i] = completion_sum
            s[i] = start_sum
            model += single_assign == 1

    if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
        for i in range(scheduling_problem.N):
            for j in range(scheduling_problem.M):
                model += x[i][j] <= scheduling_problem.U[i][j]

    z = model.add_var(name='z')
    for i in range(scheduling_problem.N):
        model += z >= C[i]

    model.objective = z

    for i in range(scheduling_problem.N):
        for j in range(scheduling_problem.N):
            if j != i:
                if not scheduling_problem.multimachine:
                    model += sigma[i][j] + sigma[j][i] == 1
                else:
                    model += sigma[i][j] + sigma[j][i] <= 1
                    model += epsilon[i][j] + epsilon[j][i] <= 1
                    model += epsilon[i][j] + epsilon[j][i] + sigma[i][j] + sigma[j][i] >= 1
                    model += p[j] - p[i] - epsilon[i][j] * (scheduling_problem.M + 1) <= 0
                    model += p[j] - p[i] - 1 - (epsilon[i][j] - 1) * (scheduling_problem.M + 1) >= 0

                model += s[j] - C[i] - (sigma[i][j] - 1)*scheduling_problem.W >= 0
                if scheduling_problem.problem_type.task_relation_type == TaskRelationType.PRECEDENCE:
                    model += sigma[i][j] >= scheduling_problem.A[i, j]

                if scheduling_problem.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
                    for h in range(scheduling_problem.M):
                        for k in range(scheduling_problem.M):
                            model += x[i][h] - gamma[i][j][h][k] >= 0
                            model += x[j][k] - gamma[i][j][h][k] >= 0
                            model += x[i][h] + x[j][k] - 1 - gamma[i][j][h][k] <= 0
                            model += scheduling_problem.A[i][j]*gamma[i][j][h][k] <= scheduling_problem.B[h][k]

    status = model.optimize()

    if status == mip.OptimizationStatus.INFEASIBLE:
        return False

    objective = model.objective_value

    schedule = []

    for i in range(scheduling_problem.N):
        if scheduling_problem.multimachine:
            schedule.append((stsum(g, i, scheduling_problem.M, scheduling_problem.num_steps, scheduling_problem.t_step),
                             ptsum(g, i, scheduling_problem.M, scheduling_problem.num_steps)))
        else:
            schedule.append((stsum(g, i, scheduling_problem.M, scheduling_problem.num_steps, scheduling_problem.t_step), 0))

    return schedule, objective

def compute_approximation_schedule(scheduling_problem, h):
    p = {}

    if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
        scheduling_problem.U = np.zeros((scheduling_problem.N, scheduling_problem.M))
        for i in range(scheduling_problem.N):
            for j in range(scheduling_problem.M):
                if scheduling_problem.task_types[i] == scheduling_problem.machine_types[j]:
                    scheduling_problem.U[i, j] = 1

    model = mip.Model(solver_name=mip.CBC)
    model.verbose = 0

    s = [model.add_var(name='s({})'.format(i + 1)) for i in range(scheduling_problem.N)]
    C = [model.add_var(name='C({})'.format(i + 1)) for i in range(scheduling_problem.N)]
    sigma = [[model.add_var(var_type=mip.BINARY, name='sigma({},{})'.format(i + 1, j + 1)) for i in range(scheduling_problem.N)]
             for j in range(scheduling_problem.N)]
    if scheduling_problem.multimachine:
        p = [model.add_var(var_type=mip.INTEGER, name='p({})'.format(i + 1)) for i in range(scheduling_problem.N)]
        x = [[model.add_var(var_type=mip.BINARY, name='x({},{}'.format(i + 1, k + 1)) for k in range(scheduling_problem.M)]
             for i in range(scheduling_problem.N)]
        epsilon = [
            [model.add_var(var_type=mip.BINARY, name='epsilon({},{})'.format(i + 1, j + 1)) for i in range(scheduling_problem.N)]
            for j in range(scheduling_problem.N)]
        proc_assign_sum = {}

    if scheduling_problem.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
        gamma = [
            [
                [
                    [
                        model.add_var(var_type=mip.BINARY, name='gamma({},{},{},{})'.format(i + 1, j + 1, h + 1, k + 1))
                        for k in range(scheduling_problem.M)
                    ]
                    for h in range(scheduling_problem.M)
                ]
                for j in range(scheduling_problem.N)
            ]
            for i in range(scheduling_problem.N)
        ]

    for i in range(scheduling_problem.N):
        model += s[i] >= 0
        model += C[i] <= scheduling_problem.W
        if scheduling_problem.multimachine:
            model += p[i] >= 0
            model += p[i] <= scheduling_problem.M - 1
            p[i] = 0
            proc_assign_sum[i] = 0
            for k in range(scheduling_problem.M):
                p[i] = p[i] + k * x[i][k]
                proc_assign_sum[i] = proc_assign_sum[i] + x[i][k]
            model += proc_assign_sum[i] == 1

        if len(scheduling_problem.delta_sample.shape) == 1:
            C[i] = scheduling_problem.h[0] * s[i] + scheduling_problem.h[1]
        elif len(scheduling_problem.delta_sample.shape) == 2:
            C[i] = scheduling_problem.h[i, 0] * s[i] + scheduling_problem.h[i, 1]
        # TODO:  change to check for specifically het machines vs het tasks, the above is only valid for tasks
        elif len(scheduling_problem.delta_sample.shape) == 3:
            if scheduling_problem.het_method_hyperplane == 0:
                C[i] = scheduling_problem.h[i, 0] * s[i] + scheduling_problem.h[i, 1] * p[i] + scheduling_problem.h[i, 2]
            elif scheduling_problem.het_method_hyperplane == 1:
                C[i] = scheduling_problem.h_1[i] * s[i]
                for k in range(scheduling_problem.M):
                    C[i] = C[i] + x[i][k] * scheduling_problem.h_2[i, k]

    if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
        for i in range(scheduling_problem.N):
            for j in range(scheduling_problem.M):
                model += x[i][j] <= scheduling_problem.U[i][j]

    z = model.add_var(name='z')
    for i in range(scheduling_problem.N):
        model += z >= C[i]

    model.objective = z

    for i in range(scheduling_problem.N):
        for j in range(scheduling_problem.N):
            if j != i:
                if not scheduling_problem.multimachine:
                    model += sigma[i][j] + sigma[j][i] == 1
                else:
                    model += sigma[i][j] + sigma[j][i] <= 1
                    model += epsilon[i][j] + epsilon[j][i] <= 1
                    model += epsilon[i][j] + epsilon[j][i] + sigma[i][j] + sigma[j][i] >= 1
                    model += p[j] - p[i] - epsilon[i][j] * (scheduling_problem.M + 1) <= 0
                    model += p[j] - p[i] - 1 - (epsilon[i][j] - 1) * (scheduling_problem.M + 1) >= 0

                model += s[j] - C[i] - (sigma[i][j] - 1) * scheduling_problem.W >= 0
                if scheduling_problem.problem_type.task_relation_type == TaskRelationType.PRECEDENCE:
                    model += sigma[i][j] >= scheduling_problem.A[i, j]

                if scheduling_problem.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
                    for h in range(scheduling_problem.M):
                        for k in range(scheduling_problem.M):
                            model += x[i][h] - gamma[i][j][h][k] >= 0
                            model += x[j][k] - gamma[i][j][h][k] >= 0
                            model += x[i][h] + x[j][k] - 1 - gamma[i][j][h][k] <= 0
                            model += scheduling_problem.A[i][j] * gamma[i][j][h][k] <= scheduling_problem.B[h][k]

    status = model.optimize()

    if status == mip.OptimizationStatus.INFEASIBLE:
        return False

    objective = model.objective_value

    # print(scheduling_problem.objective)


    if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS and scheduling_problem.p_permuted:
        # unpermute
        pasum_series = np.zeros(scheduling_problem.N)
        for o in range(scheduling_problem.N):
            pasum_series[o] = pasum(x, i, scheduling_problem.M)
        schedule_series = pasum_series.copy()
        big_permute = []
        for u in range(scheduling_problem.num_types):
            big_permute = np.concatenate((big_permute, np.array(scheduling_problem.P_permutation[u])))

    schedule = []

    for i in range(scheduling_problem.N):
        if scheduling_problem.multimachine:
            if scheduling_problem.p_permuted:
                if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HOMOGENEOUS:
                    schedule.append((s[i].x, scheduling_problem.P_permutation[pasum(x, i, scheduling_problem.M)]))
                elif scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
                    schedule.append((s[i].x, np.where(big_permute == pasum(x, i, scheduling_problem.M))[0][0]))
            else:
                schedule.append((s[i].x, pasum(x, i, scheduling_problem.M)))
        else:
            schedule.append((s[i].x, 0))

    return schedule, objective
    # print("Schedule")
    # print(scheduling_problem.schedule)

