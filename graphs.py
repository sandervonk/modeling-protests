# https://networkx.org/documentation/stable/tutorial.html
# https://stackoverflow.com/questions/19212979/draw-graph-in-networkx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as pyplot
import networkx as nx
import random
import time
import math


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
    colors = {
        "S": "blue",    # Potential Protestors
        "I": "red",     # New Protestors
        "C": "orange",  # Mature Protestors
        "R": "grey",    # Retired Protestors
    }

    takeFrom = {
        "I": ["S"],  # New comes from potential
        "C": ["I"],  # Mature comes from new
        "R": ["I", "C"],  # Retired comes from new & mature
        "S": ["R"]  # New comes from retired
    }
    changeTo = {
        "S": "I",
        "I": "C",
        "R": "S",
        "C": "R"
    }

    def __init__(self, size, seed_verts=10, intitial_contacts=2, secondary_contacts=1):
        n, n_0, m_r, m_s = size, seed_verts, intitial_contacts, secondary_contacts

        self.size = n

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
        self.graph.add_nodes_from([(id, self.getBlankAttribute()) for id in range(n_0)])

        # Steps 2-4 (5): add in a new vertex
        while self.graph.number_of_nodes() < n:
            self.connect_new(m_r, m_s)

        # Create an associated model to provide updates based on the second paper
        self.model = Model(self)

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

        self.graph.add_nodes_from([(id, self.getBlankAttribute())])
        self.graph.add_edges_from(edges)

    def draw(self):
        nodeTuples = self.graph.nodes(data=True)
        color_map = [SocialNetwork.colors[attrs["code"]] for id, attrs in nodeTuples]

        pos = nx.kamada_kawai_layout(self.graph, scale=2)
        nx.draw_networkx(self.graph, pos, node_size=50, with_labels=False, node_color=color_map)
        pyplot.show()

    def getBlankAttribute(self):
        return {"code": "S"}

    def runDelta(self, code, delta):
        delta = round(delta)
        print(f"run d{code}={delta}")
        if delta == 0:  # no change
            return
        elif delta < 0:  # remove n=delta with code
            pop = sorted(filter(lambda d: code == d[1]["code"], self.graph.nodes(data=True)))
            choice = random.sample(pop, min(len(pop), abs(delta)))
            for id, data in choice:
                data["code"] = SocialNetwork.changeTo[code]
        elif delta > 0:  # add code to n=delta of type blank
            pop = sorted(filter(lambda d: d[1]["code"] in SocialNetwork.takeFrom[code], self.graph.nodes(data=True)))
            choice = random.sample(pop, min(len(pop), delta))
            for id, data in choice:
                data["code"] = code

    def runUpdate(self, update):
        props, values, deltas = update.props, update.value, update.delta
        for key in props:
            delta = deltas[key]
            self.runDelta(key, delta)

    def step(self):
        self.model.step()

    def seed(self):
        self.model.seed()


class Model:
    def __init__(self, network):
        self.network = network
        self.step_num = 0

        num = network.size     # Size of input network
        self.n = num           # Total number of population

        self.S = num           # TODO Potential Protesters (add diffusion model on top later)
        self.I = 0             # New Protesters
        self.C = 0             # Mature Protestors
        self.C_0 = self.C  # ? Initial Mature Protestors
        self.R = 0             # Retired Protestors

        self.delta_2 = 0.2   # Withdrawal rate for mature protestors
        self.delta_1 = 0.0762  # Withdrawal rate for new protestors
        self.delta_0 = self.delta_2 - self.delta_1  # Used as shorthand

        self.chi = 0.0203      # Rate of new protestors turning into mature protestors
        self.gamma = 0.03      # Rate of retired protestors turining into potential protestors

        self.beta_1 = 0.0045   # Attractiveness to become protestors from new protestors
        self.beta_2 = 0.1832   # Attractiveness to become protestors from mature protestors

    # TODO: Replace temp seeding
    def seed(self):
        self.C_0 = +10
        dS = -80
        dI = +20
        dC = self.C_0
        dR = +10
        update = ModelUpdate(self, dS, dI, dC, dR)
        self.network.runUpdate(update)
        print(self.S, self.I, self.C, self.R)

    def omega(self):
        # return self.delta_2 + self.delta_0 * (self.C_0 ** self.n) / (((self.I + self.C) ** self.n) + (self.C_0 ** self.n))
        # return (self.I + self.C) / self.network.size
        return self.delta_2 + self.delta_0 * (self.I + self.C) / self.n

    def step(self):
        omega = self.omega()
        dS = (1 if self.S <= 0 else 0)*((self.beta_1 * self.I) + (self.beta_2 * self.C)) + (self.gamma * self.R)
        dI = (0 if self.S <= 0 else 1)*((self.beta_1 * self.I) + (self.beta_2 * self.C)) - (self.chi * self.I) - (self.delta_1 * self.I)
        dC = (self.chi * self.I) - (self.C * omega)
        dR = (self.delta_1 * self.I) + (self.C * omega) - (self.gamma * self.R)

        print(dS, dI, dC, dR)

        update = ModelUpdate(self, dS, dI, dC, dR)
        self.network.runUpdate(update)
        print(self.S, self.I, self.C, self.R)


class ModelUpdate:
    def __init__(self, model, dS, dI, dC, dR):
        self.value = dict()
        self.delta = dict()
        self.props = {"S", "I", "C", "R"}

        for prop in self.props:
            self.value[prop] = getattr(model, prop)
            self.delta[prop] = eval(f"d{prop}")

            # Apply delta to internals
            setattr(model, prop, self.value[prop] + self.delta[prop])

        # Apply time step
        setattr(model, "step_num", getattr(model, "step_num") + 1)


network = SocialNetwork(200, 100)
network.seed()
while True:
    network.draw()
    network.step()
