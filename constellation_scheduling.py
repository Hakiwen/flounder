import scheduling_problem, utils
from problem_types import TaskLoadType, TaskRelationType, MachineLoadType, MachineCapabilityType, MachineRelationType, DeltaFunctionClass

import numpy as np


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

    problem_type = scheduling_problem.SchedulingProblemType(task_load_type=TaskLoadType.NONUNIFORM,
                                                            task_relation_type=TaskRelationType.PRECEDENCE,
                                                            machine_load_type=MachineLoadType.NONUNIFORM,
                                                            machine_capability_type=MachineCapabilityType.HETEROGENEOUS,
                                                            machine_relation_type=MachineRelationType.PRECEDENCE,
                                                            delta_function_class=DeltaFunctionClass.SAMPLED
                                                            )

    problem = scheduling_problem.SchedulingProblem(problem_type,
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

if __name__ == "__main__":
    S = 2

    A_S = np.zeros((S, S))
    A_S[0, 1] = 1
    # A_S[0, 2] = 1
    A_S[1, 0] = 1
    # A_S[2, 0] = 1

    H = 1000.0
    W = H/2
    t_step = 1
    # due to the way python handles this, the t_step can't be a non-float
    num_steps = int(W/t_step)
    time = np.linspace(0, W, num_steps)

    D = np.zeros((S, S, num_steps))



    D[0, 1, :] = 2 + np.sin(time)
    # D[0, 2, :] = 1 + np.cos(time)
    # D[1, 2, :] = 1
    D[1, 0, :] = D[0, 1, :]
    # D[2, 0, :] = D[0, 2, :]
    # D[2, 1, :] = D[1, 2, :]

    # Will need W to be noticably smaller than the time window
    D[0, 0, :] = 0.5*np.ones(num_steps)
    D[1, 1, :] = D[0, 0, :]

    BW = 1
    gamma = 1
    # No = 1
    # P_t = 100
    PNR = 100
    T = 3
    A_T = np.zeros((T, T))
    A_T[0, 1] = 1
    A_T[1, 2] = 1
    proc_time = np.ones(T)
    out_size = 10*np.ones(T)

    # print(D)
    # print(proc_time, out_size)
    problem = convert_constellation_scheduling(S, A_S, time, D, BW, gamma, PNR, T, A_T, proc_time, out_size, W)
    problem.plot_delta_fun()
    # problem.compute_schedule()
    # problem.WCPT_compute_schedule()
    # problem.exact_compute_schedule()
    problem.het_compute_schedule()
    print('term')
    # problem.plot_schedule()