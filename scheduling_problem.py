import utils
from problem_types import TaskLoadType, TaskRelationType, MachineType, MachineRelationType, DeltaFunctionClass

import numpy as np

from scipy import optimize

from plotly.subplots import make_subplots
from plotly import graph_objects as go
import plotly.figure_factory as ff

import mip

import random

import networkx as nx

def sin_delta_fun(t, constants):
    return constants[0]*t + constants[1]*np.sin(t) + constants[2]

def sin_delta_bar_fun(t, constants):
    return constants[0]*t + constants[1] + constants[2]

class SchedulingProblemType:
    def __init__(self,
                 task_load_type,
                 task_relation_type,
                 machine_type=MachineType.SINGLE,
                 machine_relation_type=MachineRelationType.UNRELATED,
                 delta_function_class=DeltaFunctionClass.LINESIN
                 ):

        assert isinstance(task_load_type, TaskLoadType)
        assert isinstance(task_relation_type, TaskRelationType)
        assert isinstance(machine_type, MachineType)
        assert isinstance(machine_relation_type, MachineRelationType)
        assert isinstance(delta_function_class, DeltaFunctionClass)
        self.task_load_type = task_load_type
        self.task_relation_type = task_relation_type
        self.machine_type = machine_type
        self.machine_relation_type = machine_relation_type
        self.delta_function_class = delta_function_class

class SchedulingProblem:
    def __init__(self, problem_type, N, W, delta_coeffs, A=None, M=1, B=None, t_step = 0.1):
        self.problem_type = problem_type
        assert isinstance(problem_type, SchedulingProblemType)

        self.num_delta_coeffs = 0
        if problem_type.delta_function_class == DeltaFunctionClass.LINESIN:
            self.num_delta_coeffs = 3
            self.generic_delta_fun = sin_delta_fun
            self.generic_delta_bar_fun = sin_delta_bar_fun

        self.multimachine = (problem_type.machine_type == MachineType.HOMOGENEOUS
                             or problem_type.machine_type == MachineType.HETEROGENEOUS)

        if problem_type.task_load_type == TaskLoadType.UNIFORM and not problem_type.machine_type == MachineType.HETEROGENEOUS:
            assert delta_coeffs.shape == (self.num_delta_coeffs,)
            self.delta_fun = self.delta_fun_1D
            self.delta_bar_fun = self.delta_bar_fun_1D
            self.delta_hat_fun = self.delta_hat_fun_1D

        elif problem_type.task_load_type == TaskLoadType.UNIFORM and problem_type.machine_type == MachineType.HETEROGENEOUS:
            assert delta_coeffs.shape == (M, self.num_delta_coeffs)
            self.delta_fun = self.delta_fun_2D
            self.delta_bar_fun = self.delta_bar_fun_2D
            self.delta_hat_fun = self.delta_hat_fun_2D
        elif problem_type.task_load_type == TaskLoadType.NONUNIFORM and not problem_type.machine_type == MachineType.HETEROGENEOUS:
            assert delta_coeffs.shape == (N, self.num_delta_coeffs)
            self.delta_fun = self.delta_fun_2D
            self.delta_bar_fun = self.delta_bar_fun_2D
            self.delta_hat_fun = self.delta_hat_fun_2D
        elif problem_type.task_load_type == TaskLoadType.NONUNIFORM and problem_type.machine_type == MachineType.HETEROGENEOUS:
            assert delta_coeffs.shape == (N, M, self.num_delta_coeffs)
            self.delta_fun = self.delta_fun_3D
            self.delta_bar_fun = self.delta_bar_fun_3D
            self.delta_hat_fun = self.delta_hat_fun_3D

        if A is not None:
            assert A.shape == (N, N)
            self.G = nx.to_networkx_graph(A, create_using=nx.DiGraph)
        if B is not None:
            assert B.shape == (M, M)
            self.H = nx.to_networkx_graph(B, create_using=nx.DiGraph)

        self.N = N
        self.W = W
        self.delta_coeffs = delta_coeffs
        self.A = A
        self.M = M
        self.B = B
        self.t_step = t_step

        self.delta_sample = {}
        self.delta_bar_sample = {}
        self.delta_hat_sample = {}
        self.d = {}
        self.num_steps = {}
        self.t_sample = {}
        self.schedule = {}



    def delta_fun_1D(self, t):
        return self.generic_delta_fun(t, self.delta_coeffs)

    def delta_fun_2D(self, t, i):
        return self.generic_delta_fun(t, self.delta_coeffs[i])

    def delta_fun_3D(self, t, i, j):
        return self.generic_delta_fun(t, self.delta_coeffs[i, j])

    def delta_bar_fun_1D(self, t):
        return self.generic_delta_bar_fun(t, self.delta_coeffs)

    def delta_bar_fun_2D(self, t, i):
        return self.generic_delta_bar_fun(t, self.delta_coeffs[i])

    def delta_bar_fun_3D(self, t, i, j):
        return self.generic_delta_bar_fun(t, self.delta_coeffs[i, j])

    def delta_hat_fun_1D(self, t):
        return t + self.d

    def delta_hat_fun_2D(self, t, i):
        return t + self.d[i]

    def delta_hat_fun_3D(self, t, i, k):
        return t + self.d[i, k]

    def find_WCPT(self):
        bounds = optimize.Bounds([0], [self.W])
        self.d = 1
        if len(self.delta_coeffs.shape) == 1:
            d_optimize_result = optimize.minimize(lambda t: t - self.delta_fun(t), [self.W], bounds=bounds)
            self.d = -1*d_optimize_result.fun[0]
        elif len(self.delta_coeffs.shape) == 2:
            self.d = np.zeros((self.delta_coeffs.shape[0]))
            for i in range(self.delta_coeffs.shape[0]):
                d_optimize_result = optimize.minimize(lambda t: t - self.delta_fun(t, i), [self.W], bounds=bounds)
                self.d[i] = -1*d_optimize_result.fun[0]
        elif len(self.delta_coeffs.shape) == 3:
            self.d = np.zeros((self.delta_coeffs.shape[0], self.delta_coeffs.shape[1]))
            for i in range(self.delta_coeffs.shape[0]):
                for j in range(self.delta_coeffs.shape[1]):
                    d_optimize_result = optimize.minimize(lambda t: t - self.delta_fun(t, i, j), [self.W], bounds=bounds)
                    self.d[i, j] = -1*d_optimize_result.fun[0]

    def sample_fun(self):
        if not self.d:
            self.find_WCPT()
        self.num_steps = int(self.W/self.t_step)
        self.t_sample = np.linspace(0, self.W, self.num_steps)
        if len(self.delta_coeffs.shape) == 1:
            self.delta_sample = list(map(self.delta_fun, self.t_sample))
            self.delta_bar_sample = list(map(self.delta_bar_fun, self.t_sample))
            self.delta_hat_sample = list(map(self.delta_hat_fun, self.t_sample))
        elif len(self.delta_coeffs.shape) == 2:
            self.delta_sample = np.zeros((self.delta_coeffs.shape[0], self.num_steps))
            self.delta_bar_sample = self.delta_sample.copy()
            self.delta_hat_sample = self.delta_sample.copy()
            for i in range(self.delta_coeffs.shape[0]):
                self.delta_sample[i, :] = list(map(self.delta_fun, self.t_sample, i*np.ones(self.num_steps, dtype=np.int)))
                self.delta_bar_sample[i, :] = list(map(self.delta_bar_fun, self.t_sample, i*np.ones(self.num_steps, dtype=np.int)))
                self.delta_hat_sample[i, :] = list(map(self.delta_hat_fun, self.t_sample, i*np.ones(self.num_steps, dtype=np.int)))
        elif len(self.delta_coeffs.shape) == 3:
            self.delta_sample = np.zeros((self.delta_coeffs.shape[0], self.delta_coeffs.shape[1], self.num_steps))
            self.delta_bar_sample = self.delta_sample.copy()
            self.delta_hat_sample = self.delta_sample.copy()
            for i in range(self.delta_coeffs.shape[0]):
                for j in range(self.delta_coeffs.shape[1]):
                    self.delta_sample[i, j, :] = list(map(self.delta_fun, self.t_sample, i*np.ones(self.num_steps, dtype=np.int), j*np.ones(self.num_steps, dtype=np.int)))
                    self.delta_bar_sample[i, j, :] = list(map(self.delta_bar_fun, self.t_sample, i*np.ones(self.num_steps, dtype=np.int) , j*np.ones(self.num_steps, dtype=np.int)))
                    self.delta_hat_sample[i, j, :] = list(map(self.delta_hat_fun, self.t_sample, i*np.ones(self.num_steps, dtype=np.int), j*np.ones(self.num_steps, dtype=np.int)))

    def compute_schedule(self):
        model = mip.Model(solver_name=mip.CBC)
        s = [model.add_var(name='s({})'.format(i+1)) for i in range(self.N)]
        C = [model.add_var(name='C({})'.format(i+1)) for i in range(self.N)]
        sigma = [[model.add_var(var_type=mip.BINARY, name='sigma({},{})'.format(i + 1, j + 1)) for i in range(self.N)] for j
                 in range(self.N)]
        if self.multimachine:
            p = [model.add_var(var_type=mip.INTEGER, name='p({})'.format(i + 1)) for i in range(self.N)]
            x = [[model.add_var(var_type=mip.BINARY, name='x({},{}'.format(i + 1, k + 1)) for k in range(self.M)] for i in
                 range(self.N)]
            epsilon = [
                [model.add_var(var_type=mip.BINARY, name='epsilon({},{})'.format(i + 1, j + 1)) for i in range(self.N)] for j
                in range(self.N)]
            proc_assign_sum = {}

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
            if self.problem_type.delta_function_class == DeltaFunctionClass.LINESIN:
                if len(self.delta_coeffs.shape) == 1:
                    C[i] = self.delta_coeffs[0]*s[i] + self.delta_coeffs[1] + self.delta_coeffs[2]
                elif len(self.delta_coeffs.shape) == 2:
                    C[i] = self.delta_coeffs[i, 0]*s[i] + self.delta_coeffs[i, 1] + self.delta_coeffs[i, 2]
                elif len(self.delta_coeffs.shape) == 3:
                    C[i] = self.delta_coeffs[i, p[i], 0]*s[i] + self.delta_coeffs[i, p[i], 1] + self.delta_coeffs[i, p[i], 2]

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

        model.optimize()

        def pasum(x, i):
            pasum = 0
            for k in range(self.M):
                pasum += x[i][k].x * k
            return pasum

        self.schedule = []
        for i in range(self.N):
            if self.multimachine:
                self.schedule.append((s[i].x, pasum(x, i)))
            else:
                self.schedule.append((s[i].x, 0))


    def plot_delta_fun(self):

        if not self.delta_sample:
            self.sample_fun()
        fig = go.Figure()
        if len(self.delta_coeffs.shape) == 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_sample, name=r'$\delta$'))
            fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_bar_sample, name=r'$\bar{\delta}$'))
            fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_hat_sample, name=r'$\hat{\delta}$'))
        elif len(self.delta_coeffs.shape) == 2:
            fig = make_subplots(cols=self.delta_coeffs.shape[0], rows=1)
            for i in range(self.delta_coeffs.shape[0]):
                fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_sample[i], name=r'$\delta_%i$' % (i+1)), row=1, col=i+1)
                fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_bar_sample[i], name=r'$\bar{\delta}_%i$' % (i+1)), row=1, col=i+1)
                fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_hat_sample[i], name=r'$\hat{\delta}_%i$' % (i+1)), row=1, col=i+1)
        elif len(self.delta_coeffs.shape) == 3:
            fig = make_subplots(cols=self.delta_coeffs.shape[0], rows=self.delta_coeffs.shape[1])
            for i in range(self.delta_coeffs.shape[0]):
                for j in range(self.delta_coeffs.shape[1]):
                    fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_sample[i, j], name=r'$\delta_{%i,%i$}$' % (i+1, j+1)), row=j+1,
                                  col=i+1)
                    fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_bar_sample[i, j], name=r'$\bar{\delta}_{%i,%i}$' % (i+1, j+1)),
                                  row=j+1, col=i+1)
                    fig.add_trace(go.Scatter(x=self.t_sample, y=self.delta_hat_sample[i, j], name=r'$\hat{\delta}_{%i,%i}$' % (i+1, j+1)),
                                  row=j+1, col=i+1)
        fig.update_layout(
            title="Original and Approximation Completion Time Functions",
            xaxis_title="Start Time",
            yaxis_title="Completion Time",
            height=1080,
            width=1920
        )
        fig.show()

    def plot_prec_graph(self):
        utils.plot_circ_digraph(self.G)

    def plot_schedule(self):
        fig = go.Figure()
        for i in range(self.N):
            if len(self.delta_coeffs.shape) == 1:
                fig.add_bar(x=[self.delta_fun(self.schedule[i][0]) - self.schedule[i][0]],
                            y=[self.schedule[i][1] + 1],
                            base=[self.schedule[i][0]],
                            orientation='h',
                            showlegend=True,
                            name='task %i' % (i+1)
                            )
            elif len(self.delta_coeffs.shape) == 2:
                fig.add_bar(x=[self.delta_fun(self.schedule[i][0], i) - self.schedule[i][0]],
                            y=[self.schedule[i][1] + 1],
                            base=[self.schedule[i][0]],
                            orientation='h',
                            showlegend=True,
                            name='task %i' % (i+1)
                            )

            elif len(self.delta_coeffs.shape) == 3:
                fig.add_bar(x=[self.delta_fun(self.schedule[i][0], i, self.schedule[i][1]) - self.schedule[i][0]],
                    y=[self.schedule[i][1] + 1],
                    base=[self.schedule[i][0]],
                    orientation='h',
                    showlegend=True,
                    name='task %i' % (i+1)
                    )
        fig.update_layout(
            barmode='stack',
            xaxis=dict(
                # autorange=True,
                showgrid=False
            ),
            yaxis=dict(
                # autorange=True,
                showgrid=False,
                tickformat=',d'
            ),
            title="Schedule",
            xaxis_title="Time",
            yaxis_title="Machine",
            # displymodebar=False
            height=1080,
            width=1920
            # margin=dict(
            #     l=50,
            #     r=50,
            #     b=100,
            #     t=100,
            #     pad=4
            # )
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
    W = 200

    A = np.zeros((N, N))
    A[0, 1] = 1
    A[0, 2] = 1
    A[1, 4] = 1
    A[2, 4] = 1
    A[2, 3] = 1



    problem_type_uniform = SchedulingProblemType(TaskLoadType.UNIFORM, TaskRelationType.UNRELATED)
    problem_uniform = SchedulingProblem(problem_type_uniform, N, W, delta_coeffs[0, 0, :])
    problem_uniform.plot_delta_fun()
    problem_uniform.compute_schedule()
    problem_uniform.plot_schedule()
    #
    # problem_type_nonuniform = SchedulingProblemType(TaskLoadType.NONUNIFORM, TaskRelationType.UNRELATED)
    # problem_nonuniform = SchedulingProblem(problem_type_nonuniform, N, W, delta_coeffs[:, 0, :])
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
    # problem_type_nonuniform_prec_homo = SchedulingProblemType(TaskLoadType.NONUNIFORM, TaskRelationType.PRECEDENCE, MachineType.HOMOGENEOUS)
    # problem_nonuniform_prec_homo = SchedulingProblem(problem_type_nonuniform_prec_homo, N, W, delta_coeffs[:, 0, :], A=A, M=3)
    # problem_nonuniform_prec_homo.compute_schedule()
    # problem_nonuniform_prec_homo.plot_schedule()

    problem_type_nonuniform_prec_het = SchedulingProblemType(TaskLoadType.NONUNIFORM, TaskRelationType.PRECEDENCE, MachineType.HETEROGENEOUS)
    problem_nonuniform_prec_het = SchedulingProblem(problem_type_nonuniform_prec_het, N, W, delta_coeffs, A=A, M=3)
    problem_nonuniform_prec_het.plot_delta_fun()
    problem_nonuniform_prec_het.compute_schedule()
    problem_nonuniform_prec_het.plot_schedule()
