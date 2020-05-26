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

class TaskLoadType(Enum):
    UNIFORM = 1
    NONUNIFORM = 2

class TaskRelationType(Enum):
    UNRELATED = 1
    PRECEDENCE = 2

class MachineLoadType(Enum):
    SINGLE = 1
    UNIFORM = 2
    NONUNIFORM = 3

class MachineCapabilityType(Enum):
    HOMOGENEOUS = 1
    HETEROGENEOUS = 2

class MachineRelationType(Enum):
    UNRELATED = 1
    PRECEDENCE = 2


# fun = pointer to fun e.g. delta_hat_fun_1D
# sample_basis = what to sample against in last dimension
# dim_sample = Dimension of sample in tuple, e.g. (N,M,num_steps)
def sample_generic_fun(fun, sample_basis, dim_sample):
    ones_basis = np.ones(len(sample_basis), dtype=np.int)
    ret = np.zeros(dim_sample)
    if isinstance(dim_sample, int):
        ret = np.array(list(map(fun, sample_basis)))
    elif len(dim_sample) == 2:
        for i in range(dim_sample[0]):
            ret[i, :] = list(map(fun, sample_basis, i * ones_basis))
    elif len(dim_sample) == 3:
        for i in range(dim_sample[0]):
            for j in range(dim_sample[1]):
                ret[i, j, :] = list(map(fun, sample_basis, i * ones_basis, j * ones_basis))
    return ret

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

def calculate_p_vars(p_vec, N, M):
    epsilon = np.zeros((N, N))
    x = np.zeros((N, M))

    for i in range(N):
        for j in range(N):
            if p_vec[i] < p_vec[j]:
                epsilon[i][j] = 1
        x[i][p_vec[i]] = 1
    return epsilon, x


def plot_circ_digraph(G=None, A=None):
    if G is None:
        G = nx.to_networkx_graph(A, create_using=nx.DiGraph)
    pos = nx.layout.circular_layout(G)
    node_angles = 2 * np.pi * np.linspace(0, 1, G.order(), endpoint=False) + np.pi / 2
    scale = 1
    node_x = scale * np.cos(node_angles)
    node_y = scale * np.sin(node_angles)
    trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(size=10 * np.ones(G.order())),
        text=[str(i) for i in range(1, G.order() + 1)],
        textposition="bottom center"
    )
    edge_list = list(G.edges)
    arrows = np.zeros((G.size(), 4))
    #     arrow = [x-, x+, y-, y+]
    for i in range(G.size()):
        arrows[i, 0] = node_x[edge_list[i][0]]
        arrows[i, 1] = node_x[edge_list[i][1]]
        arrows[i, 2] = node_y[edge_list[i][0]]
        arrows[i, 3] = node_y[edge_list[i][1]]

    #     print(arrows)
    fig = go.Figure(
        data=[trace],
        layout=go.Layout(
            annotations=[
                dict(
                    ax=arrows[i][0],
                    ay=arrows[i][2],
                    axref='x',
                    ayref='y',
                    x=arrows[i][1],
                    y=arrows[i][3],
                    xref='x',
                    yref='y',
                    showarrow=True,
                    arrowhead=5
                )
                for i in range(G.size())
            ],
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False
            ),
            yaxis=dict(
                autorange=True,
                showgrid=False,
                ticks='',
                showticklabels=False
            ),
            height=1080,
            width=1920
            #                        showgrid=False,
            #                        showline=False,
            #                        zeroline=False,

        )
    )
    fig.show()

# In code, first index of tasks and machines is 0
# In reference and display, first index of tasks and machines is 1


class SchedulingProblemType:
    def __init__(self,
                 task_load_type,
                 task_relation_type,
                 machine_load_type=MachineLoadType.SINGLE,
                 machine_capability_type=MachineCapabilityType.HOMOGENEOUS,
                 machine_relation_type=MachineRelationType.UNRELATED,
                 ):

        assert isinstance(task_load_type, TaskLoadType)
        assert isinstance(task_relation_type, TaskRelationType)
        assert isinstance(machine_load_type, MachineLoadType)
        assert isinstance(machine_relation_type, MachineRelationType)

        self.task_load_type = task_load_type
        self.task_relation_type = task_relation_type
        self.machine_load_type = machine_load_type
        self.machine_capability_type = machine_capability_type
        self.machine_relation_type = machine_relation_type



class SchedulingProblem:

    def __init__(self, problem_type,
                 N,
                 W,
                 delta_sample,
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
        self.delta_sample = delta_sample
        self.delta_bar_sample = None
        self.delta_hat_sample = None

        self.d = None
        self.schedule = None
        self.h = None
        self.P_perm = None
        self.p_permuted = False
        self.WCPT_schedule = None
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
            if t_step is None:
                self.t_step = np.max(np.diff(t_sample))
            self.num_steps = len(t_sample)
        else:
            self.H = self.W
            self.num_steps = 100
            # self.num_steps = int(self.W/100)
            self.t_sample = np.linspace(0, self.W, self.num_steps)

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

    def sample_delta_bar_fun(self):
        if self.h is None:
            self.approximate_delta()
        self.delta_bar_sample = sample_generic_fun(self.delta_bar_fun, self.t_sample, self.sample_dim)

    def sample_delta_hat_fun(self):
        if self.d is None:
            self.find_WCPT()
        self.delta_hat_sample = sample_generic_fun(self.delta_hat_fun, self.t_sample, self.sample_dim)

    # Takes resulting permutations and problem state to maintain guarantees
    def permute_P(self):
        assert self.P_perm is not None
        if self.problem_type.machine_capability_type == MachineCapabilityType.HOMOGENEOUS:
            if self.B is not None:
                new_B = np.zeros(self.B.shape)
                for i in range(self.M):
                    for j in range(self.M):
                        new_B[i, j] = self.B[self.P_perm[i], self.P_perm[j]]
                self.B = new_B
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

            new_delta_sample = self.delta_sample.copy()

            if self.delta_hat_sample is None:
                self.sample_delta_hat_fun()
            new_delta_hat_sample = self.delta_hat_sample.copy()

            for u in range(len(self.P_perm)):
                type_u_machines = np.where(self.machine_types == u)[0]
                num_u_machines = type_u_machines.shape[0]

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

            if self.B is not None:
                self.B = new_B

            if self.delta_sample is not None:
                self.delta_sample = new_delta_sample

        self.p_permuted = True

    # Algorithm 1 and qp
    def approximate_delta(self):

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
                    self.h[i], self.total_approx_error[i] = upperbounding_hyperplane(this_in, this_sample)

            elif self.problem_type.task_load_type == TaskLoadType.UNIFORM:
                self.h = np.zeros(h_dim)
                ds, ts = self.find_approximation_window(self.delta_sample)
                this_sample = np.append(ds, ts[-1] + self.d)
                ones = np.ones(this_sample.shape)
                this_time = np.append(ts, ts[-1])
                this_in = np.column_stack((this_time, ones))
                # this_in = this_in.transpose()
                self.h, self.total_approx_error = upperbounding_hyperplane(this_in, this_sample)

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
                            perm_h[permdex, i, :], perm_total_approx_error[permdex, i] = upperbounding_hyperplane(this_in, this_sample)
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

                        qp_res = het_qp(self.t_sample, this_B)
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
                            self.h[i, j, :], approx_error = upperbounding_hyperplane(this_in, this_sample)

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
                                perm_h[permdex, i, :], perm_total_approx_error[permdex, i] = upperbounding_hyperplane(
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

                            qp_res = het_qp(self.t_sample, this_B)
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
                            self.h[i, j, :], approx_error = upperbounding_hyperplane(this_in, this_sample)

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
        plot_circ_digraph(self.G)

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
