from .core.permute import *
from .core.sample import *
from .core.approximate import *
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
            if het_method_hyperplane == 0:
                self.delta_bar_fun = delta_bar_fun_num_0
            elif het_method_hyperplane == 1:
                self.delta_bar_fun = delta_bar_fun_num_1
            elif het_method_hyperplane == 2:
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

    def find_WCPT(self):
        self.d = find_WCPT(self.delta_sample, self.t_sample, self.H)

    # Algorithm 1 and qp
    def approximate_delta(self):

        # if self.d is None:
        #     self.d = find_WCPT(self.delta_sample, self.t_sample, self.H)

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
        # if self.d is None:
        #     self.d = find_WCPT(self.delta_sample, self.t_sample, self.H)

        self.WCPT_schedule, self.WCPT_objective = compute_WCPT_schedule(self, self.d)

    def exact_compute_schedule(self):
        self.exact_schedule, self.exact_objective = compute_exact_schedule(self)

    def compute_schedule(self):
        # if self.h is None:
        self.approximate_delta()

        self.schedule, self.objective = compute_approximation_schedule(self, self.h)



# S is the number of satellites
# A_S is the adjaceny matrix describing the undirected graph of intersatellite links
# time is the list of time that the time based values are sampled at , starting at -1 (effectively over which we can schedule)
# D describes the inter-satellite distance evolution with time
# B, gamma, No, and P_t are uniform channel capacity parameters for the inter-satellite links
# owner is the owning node of the overall application

# T is the number of tasks
# A_T is the incidence matrix describing the DAG of task precedence
# proc_time is the processing time of each task
# NOT IMPLEMENTED in_size is the input size of each task
# out_size is the output size of each task

def convert_constellation_scheduling(S, A_S, time, D,  BW, gamma, PNR, T, A_T, proc_time, out_size, W):
    assert S >= 1
    assert A_S.shape == (S, S)
    assert len(time) >= 1
    num_steps = len(time)
    time = np.array(time)
    t_step = np.max(np.diff(time))
    assert D.shape == (S, S, time.shape[0])
    assert BW >= 0
    assert gamma >= 1
    assert PNR >= 1
    # assert P_t >= 0
    assert T >= 1
    assert A_T.shape == (T, T)
    assert proc_time.shape == (T,)
    # assert in_size.shape == (T,)
    assert out_size.shape == (T,)

    # Time should be in ascending order, can check with a diff and correct with sort and resort others with result
    if time[0] > 0:
        time = time - time[0]


    # W = np.median(time)

    num_links = np.count_nonzero(np.triu(A_S))
    edge_list = []
    for i in range(S):
        for j in range(i, S):
            if A_S[i, j] == 1:
                edge_list.append([i, j])
    edge_list = np.array(edge_list)

    M = S + num_links

    B = np.zeros((M, M))
    for i in range(edge_list.shape[0]):
        B[edge_list[i][0], S + i] = 1
        B[S + i, edge_list[i][0]] = 1
        B[edge_list[i][1], S + i] = 1
        B[S + i, edge_list[i][1]] = 1

    # B += np.eye(M)
    machine_types = np.zeros(M)
    machine_types[S:] = 1

    num_outs = int(np.sum(A_T))
    out_list = []
    for i in range(T):
        for j in range(T):
            if A_T[i, j] == 1:
                out_list.append([i, j])
    out_list = np.array(out_list)


    N = T + num_outs
    A = np.zeros((N, N))
    for i in range(num_outs):
        A[out_list[i][0], T + i] = 1
        A[T + i, out_list[i][1]] = 1

    delta_sample = np.zeros((N, M, num_steps))

    task_types = np.zeros(N)
    task_types[T:] = 1

    for i in range(T):
        delta_sample[i, 0:S, :] = time + proc_time[i]

    for i in range(num_outs):
        for j in range(edge_list.shape[0]):
            capacity = BW*np.log(
                1 + np.divide(1.0, np.power(D[edge_list[j][0], edge_list[j][1], :], gamma))*(PNR)
            )
            capacity_sum = np.zeros((num_steps, num_steps))
            for t_0 in range(num_steps -1):
                for t_1 in range(t_0, num_steps):
                    capacity_sum[t_0, t_1] = np.sum(capacity[t_0: t_1])*t_step
            this_out_size = out_size[out_list[i][0]]
            transmitted_sum = np.divide(capacity_sum, this_out_size)
            for t in range(num_steps):
                index_arr = np.where(transmitted_sum[t, :] >= 1)[0]
                if index_arr.size != 0:
                    delta_sample[T + i, S + j, t] = time[np.min(index_arr)]
                else:
                    # Not strictly true, but in the terms of the schedule output, this is equivalent to any return time greater than 0
                    # delta_sample[T + i, S + j, t] = time[-1] + np.finfo(np.float64).eps
                    delta_sample[T + i, S + j, t] = time[-1] + 1

    # TODO: Implement owner

    # delta_sample = delta_sample[:, :, 0:np.min(np.where(time >= W)[0])]

    problem_type = SchedulingProblemType(task_load_type=TaskLoadType.NONUNIFORM,
                                                            task_relation_type=TaskRelationType.PRECEDENCE,
                                                            machine_load_type=MachineLoadType.NONUNIFORM,
                                                            machine_capability_type=MachineCapabilityType.HETEROGENEOUS,
                                                            machine_relation_type=MachineRelationType.PRECEDENCE
                                                            )

    problem = SchedulingProblem(problem_type,
                                                   N,
                                                   W,
                                                   delta_sample=delta_sample,
                                                   A=A,
                                                   M=M,
                                                   B=B,
                                                   t_sample=time,
                                                   task_types=task_types,
                                                   machine_types=machine_types,
                                                   het_method_hyperplane=2)

    return problem
