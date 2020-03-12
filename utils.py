import numpy as np

import mip

import plotly.graph_objects as go

import networkx as nx

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

def plot_circ_digraph(G):
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
