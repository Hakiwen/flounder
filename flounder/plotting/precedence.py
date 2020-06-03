import numpy as np
import networkx as nx
from plotly import graph_objects as go

# Plots a digraph with nodes in a circle
# G is a netwokx digraph
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

# Convert adjacency matrix to networkx graph first
def plot_prec_graph(A):
    G = nx.to_networkx_graph(A, create_using=nx.DiGraph)
    plot_circ_digraph(G)
