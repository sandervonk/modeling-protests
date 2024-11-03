# https://networkx.org/documentation/stable/tutorial.html
# https://stackoverflow.com/questions/19212979/draw-graph-in-networkx
import networkx as nx
import random
import time
import matplotlib.pyplot as pyplot
from networkx.drawing.nx_agraph import graphviz_layout


# 1 - randomly generate a social network
# 2 - combine difficusion and broadcasting model to represent flow of ideas
# 3 - potential protesters, turnout; spread of how protests are forming withing the network
# 4 - repeat 2nd and 3rd step to simulate spread

class Person:
    def __init__(self, id, membership=[], hasInfo=[]):
        self.id = id
        self.memberOf = set(membership)
        self.hasInfo = set(hasInfo)


class SocialNetwork:
    def __init__(self, size, seed_verts=10, intitial_contacts=2, secondary_contacts=1):
        n, n_0, m_r, m_s = size, seed_verts, intitial_contacts, secondary_contacts

        if not (type(n_0) == type(m_r) == type(m_s) == int):
            raise TypeError("n_0, m_r, m_s must be integers")
        elif (m_r < 1) or (m_s < 0):
            raise ValueError("m_r must be >= 1 and m_s >= 0")
        elif n_0 < 1:
            raise ValueError("n_0 verticies in seed network must be >= 1")

        # Base graph on social network model algorithm from 2.2 in
        # https://www.sciencedirect.com/science/article/pii/S0378437106003931
        self.graph = nx.Graph()

        # Step 1: start with a seed network of n_0 vertices
        self.graph.add_nodes_from(range(n_0))

        # Steps 2-4 (5): add in a new vertex
        while self.graph.number_of_nodes() < n:
            self.connect_new(m_r, m_s)

    def connect_new(self, m_r, m_s):
        id = self.graph.number_of_nodes()

        initials = random.sample(list(self.graph.nodes()), min(id, m_r))
        secondaries = []
        for initial in initials:
            neighbors = self.graph.neighbors(initial)
            secondaries.extend(random.sample(
                list(neighbors), min(len(list(neighbors)), m_s)))

        connect_to = set(initials + secondaries)
        edges = [(id, to) for to in connect_to]

        self.graph.add_node(id)
        self.graph.add_edges_from(edges)

    def draw(self):
        pos = nx.kamada_kawai_layout(self.graph, scale=2)
        nx.draw_networkx(self.graph, pos, node_size=50, with_labels=False)
        pyplot.show()


while True:
    SocialNetwork(200, 100).draw()
