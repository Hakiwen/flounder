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


def plot_schedule(schedule_to_plot, delta_sample, N, t_sample):
    fig = go.Figure()
    for i in range(N):
        if len(delta_sample.shape) == 1:
            fig.add_bar(x=[
                delta_sample[(np.abs(t_sample - schedule_to_plot[i][0]).argmin())] - schedule_to_plot[i][0]],
                # y=[schedule_to_plot[i][1] + 1],
                # y=[[schedule_to_plot[i][1] + 1], [schedule_to_plot[i][1] + 1]],
                y=[['Machine %i ' % (schedule_to_plot[i][1] + 1)], ['Task %i' % (i + 1)]],
                base=[schedule_to_plot[i][0]],
                orientation='h',
                showlegend=True,
                name='Task %i' % (i + 1)
            )
        elif len(delta_sample.shape) == 2:
            fig.add_bar(x=[
                delta_sample[i, (np.abs(t_sample - schedule_to_plot[i][0]).argmin())] - schedule_to_plot[i][
                    0]],
                # y=[[schedule_to_plot[i][1] + 1], [schedule_to_plot[i][1] + 1]],
                y=[['Machine %i ' % (schedule_to_plot[i][1] + 1)], ['Task %i' % (i + 1)]],
                base=[schedule_to_plot[i][0]],
                orientation='h',
                # showlegend=True,
                name='Task %i' % (i + 1)
            )

        elif len(delta_sample.shape) == 3:
            fig.add_bar(x=[delta_sample[
                               i, schedule_to_plot[i][1], (np.abs(t_sample - schedule_to_plot[i][0])).argmin()] -
                           schedule_to_plot[i][0]],
                        # y=[schedule_to_plot[i][1] + 1],
                        # y=[[schedule_to_plot[i][1] + 1], [schedule_to_plot[i][1] + 1]],
                        y=[['Machine %i ' % (schedule_to_plot[i][1] + 1)], ['Task %i' % (i + 1)]],
                        base=[schedule_to_plot[i][0]],
                        orientation='h',
                        showlegend=True,
                        name='Task %i' % (i + 1)
                        )
    fig.update_layout(
        # barmode='stack',
        xaxis=dict(
            # autorange=True,
            showgrid=False
            # range=[0, W]
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
    # fig.update_yaxes(range=[0, M + 1])
    # fig.update_xaxes(range=[0, W])
    fig.show(renderer="notebook")
