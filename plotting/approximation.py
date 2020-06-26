import numpy as np

from plotly.subplots import make_subplots
from plotly import graph_objects as go

from ..core.types import *

def plot_delta_fun(scheduling_problem):

    assert(scheduling_problem.delta_sample is not None)
    assert(scheduling_problem.delta_bar_sample is not None)
    assert(scheduling_problem.delta_hat_sample is not None)

    # if scheduling_problem.delta_sample is None:
    #     scheduling_problem.sample_delta_fun()
    #
    # if scheduling_problem.delta_bar_sample is None:
    #     scheduling_problem.sample_delta_bar_fun()
    #
    #
    # if scheduling_problem.delta_hat_sample is None:
    #         scheduling_problem.sample_delta_hat_fun()

    fig = go.Figure()
    if len(scheduling_problem.delta_sample.shape) == 1:
        fig = go.Figure()
        # fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_sample, name=r'$\delta$'))
        # fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_bar_sample, name=r'$\bar{\delta}$'))
        # fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_hat_sample, name=r'$\hat{\delta}$'))
        fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_sample, name=r'Exact'))
        fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_hat_sample, name=r'WCPT'))
        fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_bar_sample, name=r'Affine'))
    elif len(scheduling_problem.delta_sample.shape) == 2:
        fig = make_subplots(rows=scheduling_problem.N, cols=scheduling_problem.M)
        for i in range(scheduling_problem.delta_sample.shape[0]):
            fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_sample[i], name=r'$\delta_%i$' % (i+1)), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_bar_sample[i], name=r'$\bar{\delta}_%i$' % (i+1)), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_hat_sample[i], name=r'$\hat{\delta}_%i$' % (i+1)), row=i+1, col=1)
    elif len(scheduling_problem.delta_sample.shape) == 3:
        if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
            if scheduling_problem.num_types == 2:
            # if False:
                # for u in range(scheduling_problem.num_types):
                type_0_machines = np.where(scheduling_problem.machine_types == 0)[0]
                num_0_machines = type_0_machines.shape[0]
                type_0_tasks = np.where(scheduling_problem.task_types == 0)[0]
                num_0_tasks = type_0_tasks.shape[0]

                fig_0 = make_subplots(rows=num_0_tasks, cols=num_0_machines)

                for i in range(num_0_tasks):
                    for j in range(num_0_machines):
                        fig_0.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_sample[type_0_tasks[i], type_0_machines[j]], name=r'$\delta_{%i,%i}$' % (type_0_tasks[i] + 1, type_0_machines[j] + 1)),row=i+1, col=j+1)
                        fig_0.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_bar_sample[type_0_tasks[i], type_0_machines[j]], name=r'$\bar{\delta}_{%i,%i}$' % (type_0_tasks[i] + 1, type_0_machines[j] + 1)),row=i+1, col=j+1)
                        fig_0.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_hat_sample[type_0_tasks[i], type_0_machines[j]], name=r'$\hat{\delta}_{%i,%i}$' % (type_0_tasks[i] + 1, type_0_machines[j] + 1)),row=i+1, col=j+1)


                type_1_machines = np.where(scheduling_problem.machine_types == 1)[0]
                num_1_machines = type_1_machines.shape[0]
                type_1_tasks = np.where(scheduling_problem.task_types == 1)[0]
                num_1_tasks = type_1_tasks.shape[0]

                fig_1 = make_subplots(rows=num_1_tasks, cols=num_1_machines)

                for i in range(num_1_tasks):
                    for j in range(num_1_machines):
                        fig_1.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_sample[type_1_tasks[i], type_1_machines[j]], name=r'$\delta_{%i,%i}$' % (type_1_tasks[i] + 1, type_1_machines[j] + 1)),row=i+1, col=j+1)
                        fig_1.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_bar_sample[type_1_tasks[i], type_1_machines[j]], name=r'$\bar{\delta}_{%i,%i}$' % (type_1_tasks[i] + 1, type_1_machines[j] + 1)),row=i+1, col=j+1)
                        fig_1.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_hat_sample[type_1_tasks[i], type_1_machines[j]], name=r'$\hat{\delta}_{%i,%i}$' % (type_1_tasks[i] + 1, type_1_machines[j] + 1)),row=i+1, col=j+1)

        else:
            fig = make_subplots(cols=scheduling_problem.M, rows=scheduling_problem.N)
            for i in range(scheduling_problem.delta_sample.shape[0]):
                for j in range(scheduling_problem.delta_sample.shape[1]):
                    # fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_sample[i, j],
                    #                          name=r'$\delta_{%i,%i}$' % (i + 1, j + 1)), row=i + 1,
                    #               col=j + 1)
                    # fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_bar_sample[i, j],
                    #                          name=r'$\bar{\delta}_{%i,%i}$' % (i + 1, j + 1)),
                    #               row=i + 1, col=j + 1)
                    # fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_hat_sample[i, j],
                    #                          name=r'$\hat{\delta}_{%i,%i}$' % (i + 1, j + 1)),
                    #               row=i + 1, col=j + 1)
                    fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_sample[i, j],
                                             name=r'$\delta_{%i,%i}$' % (i + 1, j + 1)), row=i + 1,
                                  col=j + 1)
                    fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_bar_sample[i, j],
                                             name=r'$\bar{\delta}_{%i,%i}$' % (i + 1, j + 1)),
                                  row=i + 1, col=j + 1)
                    fig.add_trace(go.Scatter(x=scheduling_problem.t_sample, y=scheduling_problem.delta_hat_sample[i, j],
                                             name=r'$\hat{\delta}_{%i,%i}$' % (i + 1, j + 1)),
                                  row=i + 1, col=j + 1)
    fig.update_layout(
        # title="Original and Approximation Completion Time Functions"
        # xaxis_title="Start Time",
        xaxis_title=r"$t$"
        # height=720*scheduling_problem.N,
        # width=1280*scheduling_problem.M
    )
    if scheduling_problem.problem_type.machine_capability_type == MachineCapabilityType.HETEROGENEOUS:
        if scheduling_problem.num_types == 2:
        # if False:

            fig_0.update_layout(
                title="Original and Approximation Completion Time Functions",
                xaxis_title="Start Time",
                yaxis_title="Completion Time",
                height=1080,
                width=1920,
                font=dict(
                    family="Latin Modern",
                    size=20
                )
            )
            fig_1.update_layout(
                title="Original and Approximation Completion Time Functions",
                xaxis_title="Start Time",
                yaxis_title="Completion Time",
                height=1080,
                width=1920,
                font=dict(
                    family="Latin Modern",
                    size=20
                )
            )
        fig_0.show()
        fig_1.show()
        # fig.show()
    else:
        fig.update_layout(
            xaxis_title="Start Time",
            yaxis_title="Completion Time",
            font=dict(
                family="Latin Modern",
                size=20
            )
        )
        fig.show()
