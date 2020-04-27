import utils
from problem_types import TaskLoadType, TaskRelationType, MachineLoadType, MachineCapabilityType, MachineRelationType, DeltaFunctionClass

import numpy as np

from scipy import optimize

import itertools

import mip

from plotly.subplots import make_subplots
from plotly import graph_objects as go
import plotly.figure_factory as ff

import networkx as nx


def sin_delta_fun(t, constants):
    return constants[0]*t + constants[1]*np.sin(t) + constants[2]

def sin_delta_bar_fun(t, constants):
    return constants[0]*t + constants[1] + constants[2]

# In code, first index of tasks and machines is 0
# In reference and display, first index of tasks and machines is 1


class SchedulingProblemType:
    def __init__(self,
                 task_load_type,
                 task_relation_type,
                 machine_load_type=MachineLoadType.SINGLE,
                 machine_capability_type=MachineCapabilityType.HOMOGENEOUS,
                 machine_relation_type=MachineRelationType.UNRELATED,
                 delta_function_class=DeltaFunctionClass.LINESIN
                 ):

        assert isinstance(task_load_type, TaskLoadType)
        assert isinstance(task_relation_type, TaskRelationType)
        assert isinstance(machine_load_type, MachineLoadType)
        assert isinstance(machine_relation_type, MachineRelationType)
        assert isinstance(delta_function_class, DeltaFunctionClass)
        self.task_load_type = task_load_type
        self.task_relation_type = task_relation_type
        self.machine_load_type = machine_load_type
        self.machine_capability_type = machine_capability_type
        self.machine_relation_type = machine_relation_type
        self.delta_function_class = delta_function_class


class SchedulingProblem:

    def __init__(self, problem_type,
                 N,
                 W,
                 delta_sample=None,
                 delta_coeffs=None,
                 A=None,
                 M=1,
                 B=None,
                 t_step=None,
                 task_types=None,
                 machine_types=None,
                 t_sample=None,
                 het_method_hyperplane=2
                 ):
        self.problem_type = problem_type
        assert isinstance(problem_type, SchedulingProblemType)

        self.het_method_hyperplane = het_method_hyperplane
        self.delta_sample = None
        self.delta_bar_sample = None
        self.delta_hat_sample = None
        self.delta_coeffs = None
        self.d = None
        self.schedule = None
        self.h = None
        self.P_perm = None
        self.p_permuted = False
        self.WCPT_schedule = None
        self.schedule = None
        self.H = None

        self.N = N
        self.W = W

        self.A = A
        self.M = M
        self.B = B
        self.t_sample = t_sample
        self.t_step = t_step
        if t_sample is not None:
            self.H = t_sample[-1]
            if t_step == None:
                self.t_step = np.max(np.diff(t_sample))
            self.num_steps = len(t_sample)
        else:
            self.H = self.W
            self.num_steps = 100
            # self.num_steps = int(self.W/100)
            self.t_sample = np.linspace(0, self.W, self.num_steps)

        # TODO: change numbers to big decimal or something similar



        if problem_type.delta_function_class == DeltaFunctionClass.LINESIN:
            self.num_delta_coeffs = 0
            self.delta_coeffs = delta_coeffs
            assert delta_coeffs is not None
            self.num_delta_coeffs = 3
            self.generic_delta_fun = sin_delta_fun
            self.generic_delta_bar_fun = sin_delta_bar_fun
            if problem_type.task_load_type == TaskLoadType.UNIFORM and not problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                assert delta_coeffs.shape == (self.num_delta_coeffs,)
                self.delta_fun = self.delta_fun_1D
                self.delta_bar_fun = self.delta_bar_fun_1D
                self.delta_hat_fun = self.delta_hat_fun_1D
                self.sample_dim = (self.num_steps,)
            elif problem_type.task_load_type == TaskLoadType.UNIFORM and problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                assert delta_coeffs.shape == (M, self.num_delta_coeffs)
                self.delta_fun = self.delta_fun_2D
                self.delta_bar_fun = self.delta_bar_fun_2D
                self.delta_hat_fun = self.delta_hat_fun_2D
                self.sample_dim = (M, self.num_steps)
            elif problem_type.task_load_type == TaskLoadType.NONUNIFORM and not problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                assert delta_coeffs.shape == (N, self.num_delta_coeffs)
                self.delta_fun = self.delta_fun_2D
                self.delta_bar_fun = self.delta_bar_fun_2D
                self.delta_hat_fun = self.delta_hat_fun_2D
                self.sample_dim = (N, self.num_steps)
            elif problem_type.task_load_type == TaskLoadType.NONUNIFORM and problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                assert delta_coeffs.shape == (N, M, self.num_delta_coeffs)
                self.delta_fun = self.delta_fun_3D
                self.delta_bar_fun = self.delta_bar_fun_3D
                self.delta_hat_fun = self.delta_hat_fun_3D
                self.sample_dim = (N, M, self.num_steps)

        elif problem_type.delta_function_class == DeltaFunctionClass.SAMPLED:
            assert delta_sample is not None
            self.delta_sample = delta_sample
            if problem_type.task_load_type == TaskLoadType.UNIFORM and not problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                assert delta_sample.shape == (self.num_steps,)
                self.delta_bar_fun = self.delta_bar_fun_1D
                self.delta_hat_fun = self.delta_hat_fun_1D
                self.sample_dim = (self.num_steps)

            elif problem_type.task_load_type == TaskLoadType.UNIFORM and problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                assert delta_sample.shape == (M, self.num_steps)
                self.delta_bar_fun = self.delta_bar_fun_2D
                self.delta_hat_fun = self.delta_hat_fun_2D
                self.sample_dim = (M, self.num_steps)

            elif problem_type.task_load_type == TaskLoadType.NONUNIFORM and not problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                assert delta_sample.shape == (N, self.num_steps)
                self.delta_bar_fun = self.delta_bar_fun_2D
                self.delta_hat_fun = self.delta_hat_fun_2D
                self.sample_dim = (N, self.num_steps)

            elif problem_type.task_load_type == TaskLoadType.NONUNIFORM and problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                assert delta_sample.shape == (N, M, self.num_steps)
                self.delta_bar_fun = self.delta_bar_fun_3D
                self.delta_hat_fun = self.delta_hat_fun_3D
                self.sample_dim = (N, M, self.num_steps)

        self.multimachine = (problem_type.machine_load_type == MachineLoadType.UNIFORM
                             or problem_type.machine_load_type == MachineLoadType.NONUNIFORM)

        if problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            self.task_types = np.array(task_types)
            self.machine_types = np.array(machine_types)
            assert np.array_equal(np.unique(task_types), np.unique(machine_types))
            self.num_types = np.unique(task_types).shape[0]

        if A is not None:
            assert A.shape == (N, N)
            self.G = nx.to_networkx_graph(A, create_using=nx.DiGraph)
        if B is not None:
            assert B.shape == (M, M)
            self.J = nx.to_networkx_graph(B, create_using=nx.DiGraph)


    def delta_fun_1D(self, t):
        return self.generic_delta_fun(t, self.delta_coeffs)

    def delta_fun_2D(self, t, i):
        return self.generic_delta_fun(t, self.delta_coeffs[i])

    def delta_fun_3D(self, t, i, j):
        return self.generic_delta_fun(t, self.delta_coeffs[i, j])

    def delta_bar_fun_1D(self, t):
        return self.h[0]*t + self.h[1]

    def delta_bar_fun_2D(self, t, i):
        return self.h[i, 0]*t + self.h[i, 1]

    # Assumes second dimension corresponds to k or another in dimension for the mapping
    def delta_bar_fun_3D(self, t, i, j):
        if self.het_method_hyperplane == 0:
            return self.h[i, 0]*t + self.h[i, 1]*j + self.h[i, 2]
        elif self.het_method_hyperplane == 1:
            return self.h_1[i]*t + self.h_2[i, j]
        elif self.het_method_hyperplane == 2:
            return self.h[i, j, 0]*t + self.h[i, j, 1]

    def delta_hat_fun_1D(self, t):
        return t + self.d

    def delta_hat_fun_2D(self, t, i):
        return t + self.d[i]

    def delta_hat_fun_3D(self, t, i, k):
        return t + self.d[i, k]

    def find_approximation_window(self, delta_sample):
        ds_index = (delta_sample <= self.H)
        ds = delta_sample[ds_index]
        ts = self.t_sample[ds_index]
        return ds, ts

    def find_WCPT(self):
        bounds = optimize.Bounds([0], [self.W])
        self.d = -1
        if self.delta_sample is None:
            self.sample_delta_fun()
        if len(self.delta_sample.shape) == 1:
            ds, ts = self.find_approximation_window(self.delta_sample)
            if ds.shape[0] > 0:
                self.d = np.max(ds - ts)
            else:
                self.d = self.H + 1

        elif len(self.delta_sample.shape) == 2:
            self.d = np.zeros((self.delta_sample.shape[0]))
            for i in range(self.delta_sample.shape[0]):
                ds, ts = self.find_approximation_window(self.delta_sample[i, :])
                if ds.shape[0] > 0:
                    self.d[i] = np.max(ds - ts)
                else:
                    self.d[i] = self.H + 1
        elif len(self.delta_sample.shape) == 3:
            self.d = np.zeros((self.delta_sample.shape[0], self.delta_sample.shape[1]))
            for i in range(self.delta_sample.shape[0]):
                for j in range(self.delta_sample.shape[1]):
                    ds, ts = self.find_approximation_window(self.delta_sample[i, j, :])
                    if ds.shape[0] > 0:
                        self.d[i, j] = np.max(ds - ts)
                    else:
                        # make WCPT greater than H if there exists no delta sample less than H
                        self.d[i, j] = self.H + 1

    def sample_delta_fun(self):
        self.delta_sample = utils.sample_generic_fun(self.delta_fun, self.t_sample, self.sample_dim)

    def sample_delta_bar_fun(self):
        if self.h is None:
            self.approximate_delta()
        self.delta_bar_sample = utils.sample_generic_fun(self.delta_bar_fun, self.t_sample, self.sample_dim)

    def sample_delta_hat_fun(self):
        if self.d is None:
            self.find_WCPT()
        self.delta_hat_sample = utils.sample_generic_fun(self.delta_hat_fun, self.t_sample, self.sample_dim)

    # Takes resulting permutations and problem state to maintain guarantees
    def permute_P(self):
        assert self.P_perm is not None
        if self.problem_type.machine_capability_type == MachineCapabilityType.HOMOGENEOUS:
            if self.delta_coeffs is not None:
                new_delta_coeffs = np.zeros(self.delta_coeffs.shape)
                for i in range(self.M):
                    new_delta_coeffs[:, i, :] = self.delta_coeffs[:, self.P_perm[i], :]
                self.delta_coeffs = new_delta_coeffs
            if self.B is not None:
                new_B = np.zeros(self.B.shape)
                for i in range(self.M):
                    for j in range(self.M):
                        new_B[i, j] = self.B[self.P_perm[i], self.P_perm[j]]
                self.B = new_B
            if self.delta_sample is None:
                self.sample_delta_fun()
            new_delta_sample = np.zeros(self.delta_sample.shape)
            for i in range(self.M):
                new_delta_sample[:, i, :] = self.delta_sample[:, self.P_perm[i], :]
            self.delta_sample = new_delta_sample

            if self.delta_hat_sample is None:
                self.sample_delta_hat_fun()
            new_delta_hat_sample = np.zeros(self.delta_hat_sample.shape)
            for i in range(self.M):
                new_delta_hat_sample[:, i, :] = self.delta_hat_sample[:, self.P_perm[i], :]
            self.delta_hat_sample = new_delta_hat_sample
        elif self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            if self.B is not None:
                new_B = self.B.copy()

            if self.delta_sample is None:
                self.sample_delta_fun()
            new_delta_sample = self.delta_sample.copy()

            if self.delta_hat_sample is None:
                self.sample_delta_hat_fun()
            new_delta_hat_sample = self.delta_hat_sample.copy()

            if self.delta_coeffs is not None:
                new_delta_coeffs = self.delta_coeffs.copy()

            for u in range(len(self.P_perm)):
                type_u_machines = np.where(self.machine_types == u)[0]
                num_u_machines = type_u_machines.shape[0]

                if self.delta_coeffs is not None:
                    for i in range(num_u_machines):
                        new_delta_coeffs[:, type_u_machines[i], :] = self.delta_coeffs[:, self.P_perm[u][i], :]
                    self.delta_coeffs = new_delta_coeffs

                if self.B is not None:
                    for i in range(num_u_machines):
                        for j in range(num_u_machines):
                            new_B[type_u_machines[i], type_u_machines[j]] = self.B[self.P_perm[u][i], self.P_perm[u][j]]
                    self.B = new_B

                for i in range(num_u_machines):
                    new_delta_sample[:, type_u_machines[i], :] = self.delta_sample[:, self.P_perm[u][i], :]
                self.delta_sample = new_delta_sample

                for i in range(num_u_machines):
                    new_delta_hat_sample[:, type_u_machines[i], :] = self.delta_hat_sample[:, self.P_perm[u][i], :]
                self.delta_hat_sample = new_delta_hat_sample

            if self.delta_coeffs is not None:
                self.delta_coeffs = new_delta_coeffs

            if self.B is not None:
                self.B = new_B

            if self.delta_sample is not None:
                self.delta_sample = new_delta_sample

        self.p_permuted = True



    # Algorithm 1 and qp
    def approximate_delta(self):
        if self.delta_sample is None:
            self.sample_delta_fun()

        if self.d is None:
            self.find_WCPT()

        if self.problem_type.machine_load_type == MachineLoadType.SINGLE or self.problem_type.machine_load_type == MachineLoadType.UNIFORM:
            h_dim = 2
            ones = np.ones(self.num_steps)
            # t_sample
            if self.problem_type.task_load_type == TaskLoadType.NONUNIFORM:
                self.h = np.zeros((self.N, h_dim))
                self.total_approx_error = np.zeros(self.h.shape[:-1])
                for i in range(self.N):
                    # We append the sample of the WCPT intercept at H
                    ds, ts = self.find_approximation_window(self.delta_sample[i, :])
                    this_sample = np.append(ds, ts[-1] + self.d[i])

                    ones = np.ones(this_sample.shape)
                    this_time = np.append(ts, ts[-1])
                    this_in = np.column_stack((this_time, ones))
                    self.h[i], self.total_approx_error[i] = utils.upperbounding_hyperplane(this_in, this_sample)
                    # self.h[i], self.total_approx_error = utils.qp_1(this_in, this_sample, self.d[i], self.W)

            elif self.problem_type.task_load_type == TaskLoadType.UNIFORM:
                self.h = np.zeros(h_dim)
                ds, ts = self.find_approximation_window(self.delta_sample)
                this_sample = np.append(ds, ts[-1] + self.d)
                ones = np.ones(this_sample.shape)
                this_time = np.append(ts, ts[-1])
                this_in = np.column_stack((this_time, ones))
                # this_in = this_in.transpose()
                self.h, self.total_approx_error = utils.upperbounding_hyperplane(this_in, this_sample)
                # self.h, self.total_approx_error = utils.qp_1(this_in, this_sample, self.d, self.W)

        elif self.problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
            if self.problem_type.machine_capability_type == MachineCapabilityType.HOMOGENEOUS:
                # We get to choose our specific ordering of p, so we can also optimize over the reorderings of p

                # permutation_list = list(itertools.permutations([i for i in range(self.M)]))

                # New permutation technique, reducing set of permutations to ordering of WCPT
                if self.het_method_hyperplane == 0:
                    permutation_list = []
                    for i in range(self.N):
                        permutation_list.append(np.argsort(self.d[i, :]))

                    num_perm = len(permutation_list)
                    h_dim = 3
                    perm_h = np.zeros((num_perm, self.N, h_dim))
                    # ones = np.ones(self.M*self.num_steps)

                    # self.num_steps + 1 for the additional sample that enforces the intercept constraint
                    ones = np.ones(self.M*(self.num_steps + 1))
                    perm_total_approx_error = np.zeros((num_perm, self.N))
                    for permdex in range(num_perm):
                        for i in range(self.N):
                            this_sample = []
                            this_p_index = []
                            this_t_sample = []
                            for k in range(self.M):
                                this_sample = np.concatenate((this_sample, self.delta_sample[i, k, :]))
                                this_p_index = np.concatenate((this_p_index, permutation_list[permdex][k]*np.ones(self.num_steps)))
                                this_t_sample = np.concatenate((this_t_sample, self.t_sample))

                            #     WCPT H intercept constraint sample
                                this_sample = np.append(this_sample, self.H + self.d[i, k])
                                this_p_index = np.append(this_p_index, permutation_list[permdex][k])
                                this_t_sample = np.append(this_t_sample, self.H)

                            this_in = np.column_stack((this_t_sample, this_p_index, ones))
                            # print(this_in)
                            # print(this_in.shape)
                            perm_h[permdex, i, :], perm_total_approx_error[permdex, i] = utils.upperbounding_hyperplane(this_in, this_sample)
                    perm_sum_approx_error = np.sum(perm_total_approx_error, axis=1)
                    argmin_perm = np.argmin(perm_sum_approx_error)
                    # print(permutation_list)
                    # print(perm_sum_approx_error)
                    # print(perm_h[:, 1, :])
                    self.h = perm_h[argmin_perm, :, :]
                    self.P_perm = permutation_list[argmin_perm]
                    self.permute_P()

                elif self.het_method_hyperplane == 1:
                    self.h_1 = np.zeros(self.N)
                    self.h_2 = np.zeros((self.N, self.M))
                    for i in range(self.N):
                        this_B = np.array(self.delta_sample[i, :, :])
                        this_B.reshape((self.M, self.t_sample.shape[0]))

                        qp_res = utils.het_qp(self.t_sample, this_B)
                        self.h_1[i] = qp_res[0]
                        self.h_2[i, :] = qp_res[1]

                elif self.het_method_hyperplane == 2:
                    self.h = np.zeros((self.N, self.M, 2))
                    for i in range(self.N):
                        for j in range(self.M):
                            ds = self.delta_sample[i, j, :]
                            ds_index = (ds <= self.H)
                            ds = ds[ds_index]
                            ts = self.t_sample[ds_index]
                            this_sample = np.append(ds, self.H + self.d[i, j])
                            this_time = np.append(ts, self.H)
                            ones = np.ones(this_time.shape[0])
                            this_in = np.column_stack((this_time, ones))
                            self.h[i, j, :], approx_error = utils.upperbounding_hyperplane(this_in, this_sample)

            elif self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
                if self.het_method_hyperplane == 0:
                    h_dim = 3
                    self.h = np.zeros((self.N, 3))
                    self.P_perm = []
                    for u in range(self.num_types):
                        type_u_machines = np.where(self.machine_types == u)[0]
                        num_u_machines = type_u_machines.shape[0]
                        type_u_tasks = np.where(self.task_types == u)[0]
                        num_u_tasks = type_u_tasks.shape[0]

                        permutation_list = []
                        for i in type_u_tasks:
                            permutation_list.append(type_u_machines[np.argsort(self.d[i, type_u_machines])])

                        # permutation_list = list(itertools.permutations(type_u_machines))

                        num_perm = len(permutation_list)

                        perm_h = np.zeros((num_perm, num_u_tasks, h_dim))
                        # Additional sample for each point
                        ones = np.ones(num_u_machines * (self.num_steps + 1))
                        perm_total_approx_error = np.zeros((num_perm, self.N))
                        for permdex in range(num_perm):
                            for i in range(num_u_tasks):
                                this_sample = []
                                this_p_index = []
                                this_t_sample = []
                                for k in range(num_u_machines):
                                    this_p = permutation_list[permdex][k]
                                    this_sample = np.concatenate((this_sample, self.delta_sample[type_u_tasks[i], this_p, :]))
                                    this_p_index = np.concatenate(
                                        (this_p_index, this_p * np.ones(self.num_steps)))
                                    this_t_sample = np.concatenate((this_t_sample, self.t_sample))

                                    this_sample = np.append(this_sample, self.d[type_u_tasks[i], type_u_machines[k]] + self.H)
                                    this_p_index = np.append(this_p_index, this_p)
                                    this_t_sample = np.append(this_t_sample, self.H)
                                this_in = np.column_stack((this_t_sample, this_p_index, ones))
                                # print(this_in)
                                # print(this_in.shape)
                                perm_h[permdex, i, :], perm_total_approx_error[permdex, i] = utils.upperbounding_hyperplane(
                                    this_in, this_sample)
                        perm_sum_approx_error = np.sum(perm_total_approx_error, axis=1)
                        argmin_perm = np.argmin(perm_sum_approx_error)
                        # print(permutation_list)
                        # print(perm_sum_approx_error)
                        # print(perm_h[:, 1, :])
                        self.h[type_u_tasks, :] = perm_h[argmin_perm, :, :]
                        self.P_perm.append(permutation_list[argmin_perm])
                    self.permute_P()
                elif self.het_method_hyperplane == 1:
                    self.h_1 = np.zeros(self.N)
                    self.h_2 = np.zeros((self.N, self.M))
                    for u in range(self.num_types):
                        type_u_machines = np.where(self.machine_types == u)[0]
                        num_u_machines = type_u_machines.shape[0]
                        type_u_tasks = np.where(self.task_types == u)[0]
                        num_u_tasks = type_u_tasks.shape[0]

                        for i in type_u_tasks:
                            this_B = np.array(self.delta_sample[i, type_u_machines, :])
                            this_B.reshape((num_u_machines, self.t_sample.shape[0]))

                            qp_res = utils.het_qp(self.t_sample, this_B)
                            self.h_1[i] = qp_res[0]
                            self.h_2[i, type_u_machines] = qp_res[1]
                elif self.het_method_hyperplane == 2:
                    self.h = np.zeros((self.N, self.M, 2))
                    for i in range(self.N):
                        for j in range(self.M):
                            ds = self.delta_sample[i, j, :]
                            ds_index = (ds <= self.H)
                            ds = ds[ds_index]
                            ts = self.t_sample[ds_index]
                            this_sample = np.append(ds, ts[-1] + self.d[i, j])
                            this_time = np.append(ts, ts[-1])
                            ones = np.ones(this_time.shape[0])
                            this_in = np.column_stack((this_time, ones))
                            self.h[i, j, :], approx_error = utils.upperbounding_hyperplane(this_in, this_sample)

    def WCPT_compute_schedule(self):
        if self.d is None:
            self.find_WCPT()

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            self.U = np.zeros((self.N, self.M))
            for i in range(self.N):
                for j in range(self.M):
                    if self.task_types[i] == self.machine_types[j]:
                        self.U[i, j] = 1

        model = mip.Model(solver_name=mip.CBC)

        s = [model.add_var(name='s({})'.format(i+1)) for i in range(self.N)]
        C = [model.add_var(name='C({})'.format(i+1)) for i in range(self.N)]

        sigma = [[model.add_var(var_type=mip.BINARY, name='sigma({},{})'.format(i + 1, j + 1)) for j in range(self.N)]
                 for i in range(self.N)]
        if self.multimachine:
            p = [model.add_var(var_type=mip.INTEGER, name='p({})'.format(i + 1)) for i in range(self.N)]
            x = [[model.add_var(var_type=mip.BINARY, name='x({},{}'.format(i + 1, k + 1)) for k in range(self.M)]
                 for i in range(self.N)]
            epsilon = [
                [model.add_var(var_type=mip.BINARY, name='epsilon({},{})'.format(i + 1, j + 1)) for i in range(self.N)]
                for j in range(self.N)]
            proc_assign_sum = {}


        if self.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
            gamma = [
                [
                    [
                        [
                            model.add_var(var_type=mip.BINARY, name='z({},{},{},{})'.format(i + 1, j + 1, h + 1, k + 1))
                            for k in range(self.M)
                        ]
                        for h in range(self.M)
                    ]
                    for j in range(self.N)
                ]
                for i in range(self.N)
            ]

        for i in range(self.N):
            model += s[i] >= 0
            model += C[i] <= self.W
            if self.multimachine:
                model += p[i] >= 0
                model += p[i] <= self.M - 1
                p[i] = 0
                proc_assign_sum[i] = 0
                for k in range(self.M):
                    p[i] = p[i] + k*x[i][k]
                    proc_assign_sum[i] = proc_assign_sum[i] + x[i][k]
                model += proc_assign_sum[i] == 1

            # if self.problem_type.delta_function_class == DeltaFunctionClass.LINESIN:
            if len(self.delta_sample.shape) == 1:
                C[i] = s[i] + self.d
            elif len(self.delta_sample.shape) == 2:
                C[i] = s[i] + self.d[i]
            # TODO:  change to check for specifically het machines vs het tasks, the above is only valid for tasks
            elif len(self.delta_sample.shape) == 3:
                C[i] = s[i]
                for j in range(self.M):
                    C[i] = C[i] + self.d[i, j]*x[i][j]

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            for i in range(self.N):
                for j in range(self.M):
                    model += x[i][j] <= self.U[i][j]

        z = model.add_var(name='z')
        for i in range(self.N):
            model += z >= C[i]

        model.objective = z

        for i in range(self.N):
            for j in range(self.N):
                if j != i:
                    if not self.multimachine:
                        model += sigma[i][j] + sigma[j][i] == 1
                    else:
                        model += sigma[i][j] + sigma[j][i] <= 1
                        model += epsilon[i][j] + epsilon[j][i] <= 1
                        model += epsilon[i][j] + epsilon[j][i] + sigma[i][j] + sigma[j][i] >= 1
                        model += p[j] - p[i] - epsilon[i][j] * (self.M + 1) <= 0
                        model += p[j] - p[i] - 1 - (epsilon[i][j] - 1) * (self.M + 1) >= 0

                    model += s[j] - C[i] - (sigma[i][j] - 1)*self.W >= 0
                    if self.problem_type.task_relation_type == TaskRelationType.PRECEDENCE:
                        model += sigma[i][j] >= self.A[i, j]

                    if self.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
                        for h in range(self.M):
                            for k in range(self.M):
                                model += x[i][h] - gamma[i][j][h][k] >= 0
                                model += x[j][k] - gamma[i][j][h][k] >= 0
                                model += x[i][h] + x[j][k] - 1 - gamma[i][j][h][k] <= 0
                                model += self.A[i][j]*gamma[i][j][h][k] <= self.B[h][k]

        status = model.optimize()


        if status == mip.OptimizationStatus.INFEASIBLE:
            # print("Infeasible")
            self.WCPT_objective = -1
            return False

        self.WCPT_objective = model.objective_value
        # print(self.WCPT_objective)

        def pasum(x, i):
            pasum = 0
            for k in range(self.M):
                pasum += x[i][k].x * k
            return pasum

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS and self.p_permuted:
            # unpermute
            big_permute = []
            for u in range(self.num_types):
                big_permute = np.concatenate((big_permute, np.array(self.P_perm[u])))

        self.WCPT_schedule = []

        for i in range(self.N):
            if self.multimachine:
                if self.p_permuted:
                    if self.problem_type.machine_capability_type == MachineCapabilityType.HOMOGENEOUS:
                        self.WCPT_schedule.append((s[i].x, self.P_perm[pasum(x, i)]))
                    # self.WCPT_schedule.append((s[i].x, pasum(x, i)))
                    elif self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
                        self.WCPT_schedule.append((s[i].x, np.where(big_permute == pasum(x, i))[0][0]))
                else:
                    self.WCPT_schedule.append((s[i].x, pasum(x, i)))
            else:
                self.WCPT_schedule.append((s[i].x, 0))

        # print("Schedule")
        # print(self.WCPT_schedule)

    def exact_compute_schedule(self):
        if self.h is None:
            self.approximate_delta()

        p = {}

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            self.U = np.zeros((self.N, self.M))
            for i in range(self.N):
                for j in range(self.M):
                    if self.task_types[i] == self.machine_types[j]:
                        self.U[i, j] = 1

        model = mip.Model(solver_name=mip.CBC)

        # Rectify delta funciton
        # TODO: set H < T and then afford irregular spacing
        self.delta_rect = np.zeros(self.delta_sample.shape)
        if self.multimachine:
            for i in range(self.N):
                for j in range(self.M):
                    for t in range(self.num_steps):
                        if self.problem_type.machine_load_type == MachineLoadType.UNIFORM:
                            sample = self.delta_sample[i][t]
                            self.delta_rect[i][t] = self.t_step * np.ceil(sample / self.t_step)
                        else:
                            sample = self.delta_sample[i][j][t]
                            self.delta_rect[i][j][t] = self.t_step * np.ceil(sample / self.t_step)
        else:
            for i in range(self.N):
                for t in range(self.num_steps):
                    if self.problem_type.task_load_type == TaskLoadType.UNIFORM:
                        sample = self.delta_sample[t]
                    else:
                        sample = self.delta_sample[i][t]
                    self.delta_rect[i][t] = self.t_step * np.ceil(sample / self.t_step )

        # overall assignment variable

        if self.multimachine:
            g = [[[model.add_var(var_type=mip.BINARY, name='g({},{},{})'.format(i + 1, j + 1, t + 1)) for t in
                   range(self.num_steps)]
                  for j in range(self.M)] for i in range(self.N)]
        else:
            g = [[model.add_var(var_type=mip.BINARY, name='g({},{})'.format(i + 1, t + 1)) for t in
                  range(self.num_steps)]
                 for i in range(self.N)]
        # x = sum over time for task i machine j

        s = [model.add_var(name='s({})'.format(i+1)) for i in range(self.N)]
        C = [model.add_var(name='C({})'.format(i+1)) for i in range(self.N)]

        sigma = [[model.add_var(var_type=mip.BINARY, name='sigma({},{})'.format(i + 1, j + 1)) for i in range(self.N)]
                 for j in range(self.N)]

        if self.multimachine:
            p = [model.add_var(var_type=mip.INTEGER, name='p({})'.format(i + 1)) for i in range(self.N)]
            x = [[model.add_var(var_type=mip.BINARY, name='x({},{}'.format(i + 1, k + 1)) for k in range(self.M)]
                 for i in range(self.N)]
            epsilon = [
                [model.add_var(var_type=mip.BINARY, name='epsilon({},{})'.format(i + 1, j + 1)) for i in range(self.N)]
                for j in range(self.N)]
            proc_assign_sum = {}

            for i in range(self.N):
                for j in range(self.M):
                    x_sum = 0
                    for t in range(self.num_steps):
                        x_sum = x_sum + g[i][j][t]
                    x[i][j] = x_sum

        if self.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
            gamma = [
                [
                    [
                        [
                            model.add_var(var_type=mip.BINARY, name='gamma({},{},{},{})'.format(i + 1, j + 1, h + 1, k + 1))
                            for k in range(self.M)
                        ]
                        for h in range(self.M)
                    ]
                    for j in range(self.N)
                ]
                for i in range(self.N)
            ]

        for i in range(self.N):
            model += s[i] >= 0
            model += C[i] <= self.W
            if self.multimachine:
                model += p[i] >= 0
                model += p[i] <= self.M - 1
                p[i] = 0
                proc_assign_sum[i] = 0
                for k in range(self.M):
                    p[i] = p[i] + k*x[i][k]
                    proc_assign_sum[i] = proc_assign_sum[i] + x[i][k]
                model += proc_assign_sum[i] == 1

            # if self.problem_type.delta_function_class == DeltaFunctionClass.LINESIN:
            if self.multimachine:
                completion_sum = 0
                single_assign = 0
                start_sum = 0
                if self.problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
                    for j in range(self.M):
                        for t in range(self.num_steps):
                            completion_sum = completion_sum + g[i][j][t]*self.delta_rect[i][j][t]
                            single_assign = single_assign + g[i][j][t]
                            start_sum = start_sum + g[i][j][t]*t*self.t_step
                else:
                    for j in range(self.M):
                        for t in range(self.num_steps):
                            completion_sum = completion_sum + g[i][j][t] * self.delta_rect[i][t]
                            single_assign = single_assign + g[i][j][t]
                            start_sum = start_sum + g[i][j][t] * t * self.t_step
                C[i] = completion_sum
                s[i] = start_sum
                model += single_assign == 1
            else:
                completion_sum = 0
                single_assign = 0
                start_sum = 0
                for t in range(self.num_steps):
                    completion_sum = completion_sum + g[i][t]*self.delta_rect[i][t]
                    single_assign = single_assign + g[i][t]
                    start_sum = start_sum + g[i][t]*t*self.t_step
                C[i] = completion_sum
                s[i] = start_sum
                model += single_assign == 1

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            for i in range(self.N):
                for j in range(self.M):
                    model += x[i][j] <= self.U[i][j]

        z = model.add_var(name='z')
        for i in range(self.N):
            model += z >= C[i]

        model.objective = z

        for i in range(self.N):
            for j in range(self.N):
                if j != i:
                    if not self.multimachine:
                        model += sigma[i][j] + sigma[j][i] == 1
                    else:
                        model += sigma[i][j] + sigma[j][i] <= 1
                        model += epsilon[i][j] + epsilon[j][i] <= 1
                        model += epsilon[i][j] + epsilon[j][i] + sigma[i][j] + sigma[j][i] >= 1
                        model += p[j] - p[i] - epsilon[i][j] * (self.M + 1) <= 0
                        model += p[j] - p[i] - 1 - (epsilon[i][j] - 1) * (self.M + 1) >= 0

                    model += s[j] - C[i] - (sigma[i][j] - 1)*self.W >= 0
                    if self.problem_type.task_relation_type == TaskRelationType.PRECEDENCE:
                        model += sigma[i][j] >= self.A[i, j]

                    if self.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
                        for h in range(self.M):
                            for k in range(self.M):
                                model += x[i][h] - gamma[i][j][h][k] >= 0
                                model += x[j][k] - gamma[i][j][h][k] >= 0
                                model += x[i][h] + x[j][k] - 1 - gamma[i][j][h][k] <= 0
                                model += self.A[i][j]*gamma[i][j][h][k] <= self.B[h][k]

        status = model.optimize()

        if status == mip.OptimizationStatus.INFEASIBLE:
            # print("Infeasible")
            self.exact_objective = -1
            return False

        def stsum(g, i):
            ret = 0
            if self.multimachine:
                for j in range(self.M):
                    for t in range(self.num_steps):
                       ret += g[i][j][t].x*t*self.t_step
            else:
                for t in range(self.num_steps):
                    ret += g[i][t].x*t*self.t_step
            return ret

        def pasum(g, i):
            ret = 0
            for j in range(self.M):
                for t in range(self.num_steps):
                    ret += g[i][j][t].x*j
            return ret

        self.exact_objective = model.objective_value

        self.exact_schedule = []

        for i in range(self.N):
            if self.multimachine:
                    self.exact_schedule.append((stsum(g, i), pasum(g, i)))
            else:
                self.exact_schedule.append((stsum(g, i), 0))

    def compute_schedule(self):
        if self.h is None:
            self.approximate_delta()

        p = {}

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            self.U = np.zeros((self.N, self.M))
            for i in range(self.N):
                for j in range(self.M):
                    if self.task_types[i] == self.machine_types[j]:
                        self.U[i, j] = 1

        model = mip.Model(solver_name=mip.CBC)
        s = [model.add_var(name='s({})'.format(i+1)) for i in range(self.N)]
        C = [model.add_var(name='C({})'.format(i+1)) for i in range(self.N)]
        sigma = [[model.add_var(var_type=mip.BINARY, name='sigma({},{})'.format(i + 1, j + 1)) for i in range(self.N)]
                 for j in range(self.N)]
        if self.multimachine:
            p = [model.add_var(var_type=mip.INTEGER, name='p({})'.format(i + 1)) for i in range(self.N)]
            x = [[model.add_var(var_type=mip.BINARY, name='x({},{}'.format(i + 1, k + 1)) for k in range(self.M)]
                 for i in range(self.N)]
            epsilon = [
                [model.add_var(var_type=mip.BINARY, name='epsilon({},{})'.format(i + 1, j + 1)) for i in range(self.N)]
                for j in range(self.N)]
            proc_assign_sum = {}

        if self.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
            gamma = [
                [
                    [
                        [
                            model.add_var(var_type=mip.BINARY, name='gamma({},{},{},{})'.format(i + 1, j + 1, h + 1, k + 1))
                            for k in range(self.M)
                        ]
                        for h in range(self.M)
                    ]
                    for j in range(self.N)
                ]
                for i in range(self.N)
            ]

        for i in range(self.N):
            model += s[i] >= 0
            model += C[i] <= self.W
            if self.multimachine:
                model += p[i] >= 0
                model += p[i] <= self.M - 1
                p[i] = 0
                proc_assign_sum[i] = 0
                for k in range(self.M):
                    p[i] = p[i] + k*x[i][k]
                    proc_assign_sum[i] = proc_assign_sum[i] + x[i][k]
                model += proc_assign_sum[i] == 1

            # if self.problem_type.delta_function_class == DeltaFunctionClass.LINESIN:
            if len(self.delta_sample.shape) == 1:
                C[i] = self.h[0]*s[i] + self.h[1]
            elif len(self.delta_sample.shape) == 2:
                C[i] = self.h[i, 0]*s[i] + self.h[i, 1]
            # TODO:  change to check for specifically het machines vs het tasks, the above is only valid for tasks
            elif len(self.delta_sample.shape) == 3:
                if self.het_method_hyperplane == 0:
                    C[i] = self.h[i, 0]*s[i] + self.h[i, 1]*p[i] + self.h[i, 2]
                elif self.het_method_hyperplane == 1:
                    C[i] = self.h_1[i]*s[i]
                    for k in range(self.M):
                        C[i] = C[i] + x[i][k]*self.h_2[i, k]

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            for i in range(self.N):
                for j in range(self.M):
                    model += x[i][j] <= self.U[i][j]

        z = model.add_var(name='z')
        for i in range(self.N):
            model += z >= C[i]

        model.objective = z

        for i in range(self.N):
            for j in range(self.N):
                if j != i:
                    if not self.multimachine:
                        model += sigma[i][j] + sigma[j][i] == 1
                    else:
                        model += sigma[i][j] + sigma[j][i] <= 1
                        model += epsilon[i][j] + epsilon[j][i] <= 1
                        model += epsilon[i][j] + epsilon[j][i] + sigma[i][j] + sigma[j][i] >= 1
                        model += p[j] - p[i] - epsilon[i][j] * (self.M + 1) <= 0
                        model += p[j] - p[i] - 1 - (epsilon[i][j] - 1) * (self.M + 1) >= 0

                    model += s[j] - C[i] - (sigma[i][j] - 1)*self.W >= 0
                    if self.problem_type.task_relation_type == TaskRelationType.PRECEDENCE:
                        model += sigma[i][j] >= self.A[i, j]

                    if self.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
                        for h in range(self.M):
                            for k in range(self.M):
                                model += x[i][h] - gamma[i][j][h][k] >= 0
                                model += x[j][k] - gamma[i][j][h][k] >= 0
                                model += x[i][h] + x[j][k] - 1 - gamma[i][j][h][k] <= 0
                                model += self.A[i][j]*gamma[i][j][h][k] <= self.B[h][k]

        status = model.optimize()

        if status == mip.OptimizationStatus.INFEASIBLE:
            # print("Infeasible")
            self.objective = -1
            return False

        self.objective = model.objective_value
        # print(self.objective)

        def pasum(x, i):
            pasum = 0
            for k in range(self.M):
                pasum += x[i][k].x * k
            return pasum

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS and self.p_permuted:
            # unpermute
            pasum_series = np.zeros(self.N)
            for o in range(self.N):
                pasum_series[o] = pasum(x, i)
            schedule_series = pasum_series.copy()
            big_permute = []
            for u in range(self.num_types):
                big_permute = np.concatenate((big_permute, np.array(self.P_perm[u])))

        self.schedule = []

        for i in range(self.N):
            if self.multimachine:
                if self.p_permuted:
                    if self.problem_type.machine_capability_type == MachineCapabilityType.HOMOGENEOUS:
                        self.schedule.append((s[i].x, self.P_perm[pasum(x, i)]))
                    elif self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
                        self.schedule.append((s[i].x, np.where(big_permute == pasum(x, i))[0][0]))
                else:
                    self.schedule.append((s[i].x, pasum(x, i)))
            else:
                self.schedule.append((s[i].x, 0))

        # print("Schedule")
        # print(self.schedule)

    def het_compute_schedule(self):
        if self.h is None:
            self.approximate_delta()
        # note that we draw p from 0 to M-1 here
        possible_M_assignments = [i for i in range(self.M)]
        # generate combination of multisets of length 5
        p_cand_list = itertools.product(possible_M_assignments, repeat=self.N)
        p_actual = []
        epsilon_actual = []
        x_actual = []

        def calculate_p_vars(p_vec, N, M):
            epsilon = np.zeros((N, N ))
            x = np.zeros((N, M))

            for i in range(N):
                for j in range(N):
                    if p_vec[i] < p_vec[j]:
                        epsilon[i][j] = 1
                x[i][p_vec[i]] = 1
            return epsilon, x

        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            for p_cand in p_cand_list:
                type_valid = True
                for i in range(len(p_cand)):
                    type_valid = type_valid and self.machine_types[p_cand[i]] == self.task_types[i]
                if type_valid:
                    p_actual.append(p_cand)
                    epx = calculate_p_vars(p_cand, self.N, self.M)
                    epsilon_actual.append(epx[0])
                    x_actual.append(epx[1])
        else:
            # p_cand_list
            for i in p_cand_list:
                epx = calculate_p_vars(i, self.N, self.M)
                epsilon_actual.append(epx[0])
                x_actual.append(epx[1])
                p_actual.append(i)

        W_min_actual = []
        schedule_actual = []
        for p_index in range(len(p_actual)):
            print(p_actual[p_index])
            p_res = self.restricted_p_compute_schedule(p_actual[p_index],
                                                       epsilon_actual[p_index],
                                                       x_actual[p_index]
                                                       )
            if p_res[0] >= 0:
                W_min_actual.append(p_res[0])
                schedule_actual.append(p_res[1])
                # print(p_res)

        self.W_min_actual = np.array(W_min_actual)
        self.schedule_actual = schedule_actual
        min_schedule_index = np.argmin(W_min_actual)
        self.objective = self.W_min_actual[min_schedule_index]
        self.schedule = schedule_actual[min_schedule_index]

    def restricted_p_compute_schedule(self, p, epsilon, x):


        model = mip.Model(solver_name=mip.CBC)
        s = [model.add_var(name='s({})'.format(i+1)) for i in range(self.N)]
        C = [model.add_var(name='C({})'.format(i+1)) for i in range(self.N)]
        sigma = [[model.add_var(var_type=mip.BINARY, name='sigma({},{})'.format(i + 1, j + 1)) for i in range(self.N)]
                 for j in range(self.N)]

        if self.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
            gamma = [
                [
                    [
                        [
                            model.add_var(var_type=mip.BINARY, name='gamma({},{},{},{})'.format(i + 1, j + 1, h + 1, k + 1))
                            for k in range(self.M)
                        ]
                        for h in range(self.M)
                    ]
                    for j in range(self.N)
                ]
                for i in range(self.N)
            ]

        for i in range(self.N):
            model += s[i] >= 0
            model += C[i] <= self.W
            # # TODO:  change to check for specifically het machines vs het tasks, the above is only valid for tasks
            C[i] = self.h[i, p[i], 0]*s[i] + self.h[i, p[i], 1]

        z = model.add_var(name='z')
        for i in range(self.N):
            model += z >= C[i]

        model.objective = z

        for i in range(self.N):
            for j in range(self.N):
                if j != i:
                    if not self.multimachine:
                        model += sigma[i][j] + sigma[j][i] == 1
                    else:
                        model += sigma[i][j] + sigma[j][i] <= 1
                        # model += epsilon[i][j] + epsilon[j][i] <= 1
                        model += epsilon[i][j] + epsilon[j][i] + sigma[i][j] + sigma[j][i] >= 1
                        # model += p[j] - p[i] - epsilon[i][j] * (self.M + 1) <= 0
                        # model += p[j] - p[i] - 1 - (epsilon[i][j] - 1) * (self.M + 1) >= 0

                    model += s[j] - C[i] - (sigma[i][j] - 1)*self.W >= 0
                    if self.problem_type.task_relation_type == TaskRelationType.PRECEDENCE:
                        model += sigma[i][j] >= self.A[i, j]

                    if self.problem_type.machine_relation_type == MachineRelationType.PRECEDENCE:
                        for h in range(self.M):
                            for k in range(self.M):
                                model += x[i][h] - gamma[i][j][h][k] >= 0
                                model += x[j][k] - gamma[i][j][h][k] >= 0
                                model += x[i][h] + x[j][k] - 1 - gamma[i][j][h][k] <= 0
                                model += self.A[i][j]*gamma[i][j][h][k] <= self.B[h][k]

        status = model.optimize()


        if status == mip.OptimizationStatus.INFEASIBLE:
            # print("Infeasible")
            objective = -1
            return objective, None
        else:
            objective = model.objective_value

        schedule = []

        for i in range(self.N):
            schedule.append((s[i].x, p[i]))

        return objective, schedule

    def plot_delta_fun(self):

        if self.delta_sample is None:
            self.sample_delta_fun()

        if self.delta_bar_sample is None:
            self.sample_delta_bar_fun()

        if self.delta_hat_sample is None:
            self.sample_delta_hat_fun()

        fig = go.Figure()
        if len(self.delta_sample.shape) == 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_sample, name=r'$\delta$'))
            fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_bar_sample, name=r'$\bar{\delta}$'))
            fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_hat_sample, name=r'$\hat{\delta}$'))
        elif len(self.delta_sample.shape) == 2:
            fig = make_subplots(rows=self.N, cols=self.M)
            for i in range(self.delta_sample.shape[0]):
                fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_sample[i], name=r'$\delta_%i$' % (i+1)), row=i+1, col=1)
                fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_bar_sample[i], name=r'$\bar{\delta}_%i$' % (i+1)), row=i+1, col=1)
                fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_hat_sample[i], name=r'$\hat{\delta}_%i$' % (i+1)), row=i+1, col=1)
        elif len(self.delta_sample.shape) == 3:
            if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
                if self.num_types == 2:
                # if False:
                    # for u in range(self.num_types):
                    type_0_machines = np.where(self.machine_types == 0)[0]
                    num_0_machines = type_0_machines.shape[0]
                    type_0_tasks = np.where(self.task_types == 0)[0]
                    num_0_tasks = type_0_tasks.shape[0]

                    fig_0 = make_subplots(rows=num_0_tasks, cols=num_0_machines)

                    for i in range(num_0_tasks):
                        for j in range(num_0_machines):
                            fig_0.add_trace(go.Scatter(x=self.t_sample, y=self.delta_sample[type_0_tasks[i], type_0_machines[j]], name=r'$\delta_{%i,%i}$' % (type_0_tasks[i] + 1, type_0_machines[j] + 1)),row=i+1, col=j+1)
                            fig_0.add_trace(go.Scatter(x=self.t_sample, y=self.delta_bar_sample[type_0_tasks[i], type_0_machines[j]], name=r'$\bar{\delta}_{%i,%i}$' % (type_0_tasks[i] + 1, type_0_machines[j] + 1)),row=i+1, col=j+1)
                            fig_0.add_trace(go.Scatter(x=self.t_sample, y=self.delta_hat_sample[type_0_tasks[i], type_0_machines[j]], name=r'$\hat{\delta}_{%i,%i}$' % (type_0_tasks[i] + 1, type_0_machines[j] + 1)),row=i+1, col=j+1)


                    type_1_machines = np.where(self.machine_types == 1)[0]
                    num_1_machines = type_1_machines.shape[0]
                    type_1_tasks = np.where(self.task_types == 1)[0]
                    num_1_tasks = type_1_tasks.shape[0]

                    fig_1 = make_subplots(rows=num_1_tasks, cols=num_1_machines)

                    for i in range(num_1_tasks):
                        for j in range(num_1_machines):
                            fig_1.add_trace(go.Scatter(x=self.t_sample, y=self.delta_sample[type_1_tasks[i], type_1_machines[j]], name=r'$\delta_{%i,%i}$' % (type_1_tasks[i] + 1, type_1_machines[j] + 1)),row=i+1, col=j+1)
                            fig_1.add_trace(go.Scatter(x=self.t_sample, y=self.delta_bar_sample[type_1_tasks[i], type_1_machines[j]], name=r'$\bar{\delta}_{%i,%i}$' % (type_1_tasks[i] + 1, type_1_machines[j] + 1)),row=i+1, col=j+1)
                            fig_1.add_trace(go.Scatter(x=self.t_sample, y=self.delta_hat_sample[type_1_tasks[i], type_1_machines[j]], name=r'$\hat{\delta}_{%i,%i}$' % (type_1_tasks[i] + 1, type_1_machines[j] + 1)),row=i+1, col=j+1)

            else:
                fig = make_subplots(cols=self.M, rows=self.N)
                for i in range(self.delta_sample.shape[0]):
                    for j in range(self.delta_sample.shape[1]):
                        fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_sample[i, j],
                                                 name=r'$\delta_{%i,%i}$' % (i + 1, j + 1)), row=i + 1,
                                      col=j + 1)
                        fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_bar_sample[i, j],
                                                 name=r'$\bar{\delta}_{%i,%i}$' % (i + 1, j + 1)),
                                      row=i + 1, col=j + 1)
                        fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_hat_sample[i, j],
                                                 name=r'$\hat{\delta}_{%i,%i}$' % (i + 1, j + 1)),
                                      row=i + 1, col=j + 1)
        fig.update_layout(
            # title="Original and Approximation Completion Time Functions"
            # xaxis_title="Start Time",
            xaxis_title=r"$t$"
            # height=720*self.N,
            # width=1280*self.M
        )
        if self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            if self.num_types == 2:
            # if False:
                fig_0.show()
                fig_1.show()

                fig_0.update_layout(
                    title="Original and Approximation Completion Time Functions",
                    xaxis_title="Start Time",
                    yaxis_title="Completion Time",
                    height=1080,
                    width=1920
                )
                fig_1.update_layout(
                    title="Original and Approximation Completion Time Functions",
                    xaxis_title="Start Time",
                    yaxis_title="Completion Time",
                    height=1080,
                    width=1920
                )
            # fig.show()
        else:
            fig.show()

    def plot_prec_graph(self):
        utils.plot_circ_digraph(self.G)

    def plot_schedule(self, schedule_to_plot):
        fig = go.Figure()
        for i in range(self.N):
            if len(self.delta_sample.shape) == 1:
                fig.add_bar(x=[self.delta_sample[(np.abs(self.t_sample - schedule_to_plot[i][0]).argmin())] - schedule_to_plot[i][0]],
                            # y=[schedule_to_plot[i][1] + 1],
                            # y=[[schedule_to_plot[i][1] + 1], [schedule_to_plot[i][1] + 1]],
                            y=[['Machine %i ' % (schedule_to_plot[i][1] + 1)], ['Task %i' % (i + 1)]],
                            base=[schedule_to_plot[i][0]],
                            orientation='h',
                            showlegend=True,
                            name='Task %i' % (i+1)
                            )
            elif len(self.delta_sample.shape) == 2:
                fig.add_bar(x=[self.delta_sample[i, (np.abs(self.t_sample - schedule_to_plot[i][0]).argmin())] - schedule_to_plot[i][0]],
                            # y=[[schedule_to_plot[i][1] + 1], [schedule_to_plot[i][1] + 1]],
                            y=[['Machine %i ' % (schedule_to_plot[i][1] + 1)], ['Task %i' % (i + 1)]],
                            base=[schedule_to_plot[i][0]],
                            orientation='h',
                            # showlegend=True,
                            name='Task %i' % (i+1)
                            )

            elif len(self.delta_sample.shape) == 3:
                fig.add_bar(x=[self.delta_sample[i, schedule_to_plot[i][1], (np.abs(self.t_sample - schedule_to_plot[i][0])).argmin()] - schedule_to_plot[i][0]],
                    # y=[schedule_to_plot[i][1] + 1],
                    # y=[[schedule_to_plot[i][1] + 1], [schedule_to_plot[i][1] + 1]],
                    y=[['Machine %i ' % (schedule_to_plot[i][1] + 1)], ['Task %i' % (i + 1)]],
                    base=[schedule_to_plot[i][0]],
                    orientation='h',
                    showlegend=True,
                    name='Task %i' % (i+1)
                    )
        fig.update_layout(
            # barmode='stack',
            xaxis=dict(
                # autorange=True,
                showgrid=False
                # range=[0, self.W]
            ),
            yaxis=dict(
                # autorange=True,
                showgrid=False,
                tickformat=',d'
                # showticklabels=False
            ),
            title="Schedule",
            xaxis_title="Time",
            yaxis_title="Machine",
            showlegend=False,
            # displymodebar=False
            # height=1080,
            # width=1920,
            margin=dict(
                l=0,
                r=50,
                b=100,
                t=100,
                pad=4
            )
        )
        # fig.update_yaxes(range=[0, self.M + 1])
        # fig.update_xaxes(range=[0, self.W])
        fig.show(renderer="notebook")


if __name__ == "__main__":
    # delta_coeffs = np.array([2.0, 2.0, 2.0])
    delta_coeffs = np.array([[[2., 2., 2.],
                              [1.42, 1.35, 1.89],
                              [1.41, 1.2, 1.39]],

                             [[1.8, 1.8, 1.8],
                              [1.61, 0.15, 1.45],
                              [1.46, 0.19, 1.61]],

                             [[1.6, 1.6, 1.6],
                              [1.17, 0.09, 1.33],
                              [1.15, 0.59, 1.57]],

                             [[1.4, 1.4, 1.4],
                              [1.05, 0.77, 1.34],
                              [1.34, 0.5, 1.39]],

                             [[1.2, 1.2, 1.2],
                              [1.18, 0.79, 1.15],
                              [1.13, 0.37, 1.14]]])
    N = 5
    W = 2*np.pi
    M = 3

    A = np.zeros((N, N))
    # A[0, 1] = 1
    # A[0, 2] = 1
    # A[1, 4] = 1
    # A[2, 4] = 1
    # A[2, 3] = 1
    for i in range(N - 1):
        A[i, i + 1] = 1

    # B = np.zeros((M, M))
    # B = np.eye(M)
    # B[0, 1] = 1
    # B[1, 0] = 1
    # B[0, 2] = 1
    # B[2, 0] = 1

    num_types = 2
    # 0 is proc task
    # 1 is comm task
    task_types = [0, 1, 0, 1, 0]
    machine_types = [0, 1, 0]


    # num_N = 4
    # W_max = 4*np.pi
    # num_W = 10
    # W_step = W_max/float(num_W)
    # hat_W_min = np.zeros((num_N, num_W))
    # bar_W_min = np.zeros((num_N, num_W))
    # W_l = np.linspace(W_step, W_max, num_W)
    #
    # for i in range(num_N):
    #     for j in range(len(W_l)):
    problem_type_uniform = SchedulingProblemType(TaskLoadType.UNIFORM, TaskRelationType.UNRELATED)
    problem_uniform = SchedulingProblem(problem_type_uniform, N=N, W=W, delta_coeffs=np.array([1.05, 1.0, np.pi/2]))
    # problem_uniform.plot_delta_fun()
    problem_uniform.compute_schedule()
    # problem_uniform.plot_schedule()
            # problem_uniform.WCPT_compute_schedule()
            # if problem_uniform.WCPT_schedule is not None:
            #     if len(problem_uniform.WCPT_schedule) > 0:
            #         hat_W_min[i, j] = problem_uniform.WCPT_schedule[-1][0]
            # else:
            #     hat_W_min[i, j] = -1
            #
            # if problem_uniform.schedule is not None:
            #     if len(problem_uniform.schedule) > 0:
            #         bar_W_min[i, j] = problem_uniform.schedule[-1][0]
            # else:
            #     bar_W_min[i, j] = -1
    #
    # print(hat_W_min)
    # print(bar_W_min)

    # problem_type_nonuniform = SchedulingProblemType(TaskLoadType.NONUNIFORM, TaskRelationType.UNRELATED)
    # problem_nonuniform = SchedulingProblem(problem_type_nonuniform, N=N, W=W, delta_coeffs=delta_coeffs[:, 0, :])
    # problem_nonuniform.plot_delta_fun()
    # problem_nonuniform.compute_schedule()
    # problem_nonuniform.plot_schedule()
    #
    # problem_type_nonuniform_prec = SchedulingProblemType(TaskLoadType.NONUNIFORM, TaskRelationType.PRECEDENCE)
    # problem_nonuniform_prec = SchedulingProblem(problem_type_nonuniform_prec, N, W, delta_coeffs[:, 0, :],A=A)
    # problem_nonuniform_prec.plot_prec_graph()
    # problem_nonuniform_prec.compute_schedule()
    # problem_nonuniform_prec.plot_schedule()
    #
    # problem_type_nonuniform_prec_homo = SchedulingProblemType(TaskLoadType.NONUNIFORM, TaskRelationType.PRECEDENCE, MachineLoadType.UNIFORM)
    # problem_nonuniform_prec_homo = SchedulingProblem(problem_type_nonuniform_prec_homo, N, W, delta_coeffs[:, 0, :], A=A, M=3)
    # problem_nonuniform_prec_homo.compute_schedule()
    # problem_nonuniform_prec_homo.plot_schedule()

    # problem_type_uniform = SchedulingProblemType(TaskLoadType.UNIFORM, TaskRelationType.UNRELATED)
    # problem_uniform = SchedulingProblem(problem_type_uniform, N, W, delta_coeffs=delta_coeffs[0, 0, :])
    # problem_uniform.plot_delta_fun()
    # problem_uniform.compute_schedule()
    # problem_uniform.plot_schedule()

    # problem_type_nonuniform_prec_het = SchedulingProblemType(TaskLoadType.NONUNIFORM, TaskRelationType.UNRELATED, MachineLoadType.NONUNIFORM, MachineCapabilityType.HOMOGENEOUS)
    # problem_nonuniform_prec_het = SchedulingProblem(problem_type_nonuniform_prec_het, N=N, W=W, delta_coeffs=delta_coeffs[:N, :M, :], M=M)
    # problem_nonuniform_prec_het.plot_delta_fun()
    # problem_nonuniform_prec_het.compute_schedule()
    # problem_nonuniform_prec_het.plot_schedule()

    # problem_type_nonuniform_prec_nonuniform_prec_het = SchedulingProblemType(TaskLoadType.NONUNIFORM, TaskRelationType.UNRELATED, MachineLoadType.NONUNIFORM, MachineCapabilityType.HETEROGENEOUS)
    # problem_nonuniform_prec_nonuniform_prec_het = SchedulingProblem(problem_type_nonuniform_prec_nonuniform_prec_het, N=N, W=W, delta_coeffs=delta_coeffs, M=M, task_types=task_types, machine_types=machine_types)
    # problem_nonuniform_prec_nonuniform_prec_het.plot_delta_fun()
    # problem_nonuniform_prec_nonuniform_prec_het.compute_schedule()
    # problem_nonuniform_prec_nonuniform_prec_het.WCPT_compute_schedule()

    # problem_nonuniform_prec_nonuniform_prec_het.plot_schedule()
    print('term')