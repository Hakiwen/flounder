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

from .core.permute import *
from .core.sample import *
from .core.approximate import *
from .core.types import *
from .core.schedule import *


# For method 2, calculates what the implicit epsilon constraint would be
def calculate_p_vars(p_vec, N, M):
    epsilon = np.zeros((N, N))
    x = np.zeros((N, M))

    for i in range(N):
        for j in range(N):
            if p_vec[i] < p_vec[j]:
                epsilon[i][j] = 1
        x[i][p_vec[i]] = 1
    return epsilon, x

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
        self.P_permutation = None
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
        if not problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
            if problem_type.task_load_type == TaskLoadType.UNIFORM:
                assert delta_sample.shape == (self.num_steps,)
                self.delta_bar_fun = delta_bar_fun_ut
                self.delta_hat_fun = delta_hat_fun_1D
                self.sample_dim = (self.num_steps)
            elif problem_type.task_load_type == TaskLoadType.NONUNIFORM:
                assert delta_sample.shape == (N, self.num_steps)
                self.delta_bar_fun = delta_bar_fun_nut
                self.delta_hat_fun = delta_hat_fun_2D
                self.sample_dim = (N, self.num_steps)
        else:
            # Assume any problem with nonuniform machines might as well have nonuniform tasks as well
            assert delta_sample.shape == (N, M, self.num_steps)
            self.delta_hat_fun = delta_hat_fun_3D
            self.sample_dim = (N, M, self.num_steps)
            if het_method_hyperplane == 1:
                self.delta_bar_fun = delta_bar_fun_num_0
            elif het_method_hyperplane == 2:
                self.delta_bar_fun = delta_bar_fun_num_1
            elif het_method_hyperplane == 3:
                self.delta_bar_fun = delta_bar_fun_num_2

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

    def sample_delta_bar(self):
        self.delta_bar_sample = sample_generic_fun(self.delta_bar_fun, self.h, self.t_sample, self.sample_dim)

    def sample_delta_hat(self):
        self.delta_hat_sample = sample_generic_fun(self.delta_hat_fun, self.d, self.t_sample, self.sample_dim)

    # Takes resulting permutations and problem state to maintain guarantees
    def permute_P(self):
        assert self.P_permutation is not None

        if self.delta_hat_sample is None:
            self.sample_delta_hat()

        if self.problem_type.machine_capability_type == MachineCapabilityType.HOMOGENEOUS:
            if self.B is not None:
                permute_B(self.B, self.P_permutation)

            self.delta_sample = permute_sample(self.delta_sample, self.P_permutation)
            self.delta_hat_sample = permute_sample(self.delta_hat_sample, self.P_permutation)

        elif self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:

            for u in range(len(self.P_permutation)):
                type_u_machines = np.where(self.machine_types == u)[0]

                if self.B is not None:
                    self.B = permute_typed_B(self.B, self.P_permutation[u], type_u_machines)

                self.delta_sample = permute_typed_sample(self.delta_sample, self.P_permutation[u], type_u_machines)
                self.delta_hat_sample = permute_typed_sample(self.delta_hat_sample, self.P_permutation[u], type_u_machines)

        self.p_permuted = True

    # Algorithm 1 and qp
    def approximate_delta(self):

        if self.d is None:
            self.d = find_WCPT(self.delta_sample, self.t_sample, self.H)

        # The resulting approximation on uniform machine is the same as if single machine
        if self.problem_type.machine_load_type == MachineLoadType.SINGLE or self.problem_type.machine_load_type == MachineLoadType.UNIFORM:
            if self.problem_type.task_load_type == TaskLoadType.NONUNIFORM:
                self.h, self.total_approx_error = approximate_delta_nut(self.d, self.delta_sample, self.H, self.N, self.t_sample)
            elif self.problem_type.task_load_type == TaskLoadType.UNIFORM:
                self.h, self.total_approx_error = approximate_delta_ut(self.d, self.delta_sample, self.H, self.t_sample)

        elif self.problem_type.machine_load_type == MachineLoadType.NONUNIFORM:
            if self.problem_type.machine_capability_type == MachineCapabilityType.HOMOGENEOUS:
                # We get to choose our specific ordering of p, so we can also optimize over the reorderings of p

                # permutation_list = list(itertools.permutations([i for i in range(self.M)]))

                # New permutation technique, reducing set of permutations to ordering of WCPT
                if self.het_method_hyperplane == 0:
                    self.h, self.P_permutation = approximate_delta_num_0(self.d, self.delta_sample, self.H, self.N, self.num_steps, self.M, self.t_sample)
                    self.permute_P()

                elif self.het_method_hyperplane == 1:
                    self.h = approximate_delta_num_1(self.delta_sample, self.N, self.M, self.t_sample)

                elif self.het_method_hyperplane == 2:
                    self.h, self.total_approx_error = approximate_delta_num_2(self.d, self.delta_sample, self.H, self.N, self.M, self.t_sample)

            elif self.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
                if self.het_method_hyperplane == 0:
                    self.h, self.P_permutation = approximate_delta_num_0_het(self.d, self.delta_sample, self.H, self.N, self.num_steps, self.num_types, self.M, self.machine_types, self.t_sample, self.task_types)
                    self.permute_P()
                elif self.het_method_hyperplane == 1:
                    self.h = approximate_delta_num_1_het(self.delta_sample, self.N, self.num_types, self.M, self.machine_types, self.t_sample, self.task_types)
                elif self.het_method_hyperplane == 2:
                    self.h, self.total_approx_error = approximate_delta_num_2_het(self.d, self.delta_sample, self.H, self.N, self.M, self.t_sample)

    def WCPT_compute_schedule(self):
        if self.d is None:
            self.d = find_WCPT(self.delta_sample, self.t_sample, self.H)

        self.WCPT_schedule, self.WCPT_objective = compute_WCPT_schedule(self, self.d)

    def exact_compute_schedule(self):
        self.exact_schedule, self.exact_objective = compute_exact_schedule(self)

    def compute_schedule(self):
        # if self.h is None:
        self.approximate_delta()

        self.schedule, self.objective = compute_approximation_schedule(self, self.h)

