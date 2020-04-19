import numpy as np

import mip

import plotly.graph_objects as go

import networkx as nx

# X \in R^MxN, returns hyper x \in R^N
# for writing, better definition of a hyperplane
# Also determine best objective function and complexity of this


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

# TODO: Consider reworking to produce a bar strictly less than hat
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

def qp_1(A, b, d, H):
    A = np.array(A)
    b = np.array(b)
    d = np.array(d)
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

    # if N == 2:
    #     model += H*x[0] + x[1] == H + d
    #     model += x[1] >= b[0]
    #     model += x[0] >= 1
    # elif N == 3:

    # TODO: introduce less than WCPT for multimachine cases

    total_error = mip.xsum(e[i] for i in range(M))
    model.objective = total_error
    model.optimize()
    #     x_found = np.zeros(N)
    x_found = np.array([x[i].x for i in range(N)])
    # print(x_found)
    return x_found, model.objective_value

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
