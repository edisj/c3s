import networkx as nx


def graph_reaction_network(system):
    """"""

    # directed graph
    graph = nx.DiGraph()

    nodes = system.species.copy()
    # remove substrate and product molecules from node definitions
    if 'S' in nodes:
        nodes.remove('S')
    if 'P' in nodes:
        nodes.remove('P')

