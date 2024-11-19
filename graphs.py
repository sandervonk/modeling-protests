# https://networkx.org/documentation/stable/tutorial.html
# https://stackoverflow.com/questions/19212979/draw-graph-in-networkx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as pyplot
import networkx as nx
import random
import copy
from functools import reduce

# speed up save functionality
import threading


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
    UNINFORMED = "U"
    POTENTIAL = "S"
    NEW = "I"
    MATURE = "C"
    RETIRED = "R"

    colors = {
        UNINFORMED: "grey",
        POTENTIAL: "blue",
        NEW: "red",
        MATURE: "orange",
        RETIRED: "yellow",
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
        pyplot.axis('off')

        # threaded save to reduce hanging time
        fig = pyplot.gcf()
        threading.Thread(
            target=SocialNetwork.saveFrame,
            args=(fig, self.model.step_num),
            daemon=True
        ).start()

    # thread called save implementation
    @staticmethod
    def saveFrame(fig, step_num):
        """Save the figure in a separate thread."""
        filename = f'./plot/{step_num:04d}.png'
        fig.savefig(filename, bbox_inches="tight", dpi=300)
        pyplot.close(fig)

    def getBlankAttribute(self):
        return {"code": SocialNetwork.UNINFORMED}

    def runUpdate(self, update):
        nx.set_node_attributes(self.graph, update.dump())

    def getCounts(self):
        counts = {
            SocialNetwork.UNINFORMED: 0,
            SocialNetwork.POTENTIAL: 0,
            SocialNetwork.NEW: 0,
            SocialNetwork.MATURE: 0,
            SocialNetwork.RETIRED: 0,
        }

        for id, data in self.graph.nodes(data=True):
            counts[data["code"]] += 1

        return counts

    def step(self):
        self.model.step()


class Model:
    def __init__(self, network):
        self.network = network
        self.step_num = 0

        num = network.size     # Size of input network
        self.n = num           # Total number of population

        self.delta_2 = 0.02    # Withdrawal rate for mature protestors
        self.delta_1 = 0.0762  # Withdrawal rate for new protestors
        self.delta_0 = self.delta_2 - self.delta_1  # Used as shorthand

        self.chi = 0.0203      # Rate of new protestors turning into mature protestors
        self.gamma = 0.03      # Rate of retired protestors turining into potential protestors

        self.beta_1 = 0.0450   # Attractiveness to become protestors from new protestors
        self.beta_2 = 0.1832   # Attractiveness to become protestors from mature protestors

        self.seed()

    def seed(self):
        counts = {
            SocialNetwork.UNINFORMED: 0,
            SocialNetwork.POTENTIAL: 0,
            SocialNetwork.NEW: 12,
            SocialNetwork.MATURE: 15,
            SocialNetwork.RETIRED: 15,
        }
        num = 0
        for code in counts:
            while counts[code] > 0 and num < self.n:
                self.network.graph.nodes[num]["code"] = code
                counts[code] -= 1
                num += 1

    def omega(self):
        # return self.delta_2 + self.delta_0 * (self.C_0 ** self.n) / (((self.I + self.C) ** self.n) + (self.C_0 ** self.n))
        # return (self.I + self.C) / self.network.size
        return self.delta_2 + self.delta_0 * (self.counts[SocialNetwork.NEW] + self.counts[SocialNetwork.MATURE]) / self.n

    # Apply state changes to attached network
    def step(self):
        self.counts = self.network.getCounts()

        omega = self.omega()
        update = ModelUpdate([self.getChangeState(*node, omega) for node in self.network.graph.nodes(data=True)])
        self.step_num += 1
        self.network.runUpdate(update)

    @staticmethod
    def doBinomial(n):
        p = 0.25
        # implement binominal chance as derived by Pete
        return Model.doProb(100 * (1 - (1 - p)**n))

    @staticmethod
    def doProb(chance):
        result = random.random() * 100 < abs(chance)
        return result
    # Probability-based state change, returned as the new (can be unchanged) state

    def activeNeighbors(self, id):
        count = 0
        for neighborId in self.network.graph.neighbors(id):
            neighbor = self.network.graph.nodes[neighborId]
            if neighbor["code"] != SocialNetwork.UNINFORMED:
                count += 1
        return count

    def getChangeState(self, id, data, omega):
        newData = copy.copy(data)
        match data["code"]:
            case SocialNetwork.UNINFORMED:
                if Model.doBinomial(self.activeNeighbors(id)):
                    newData["code"] = SocialNetwork.POTENTIAL
            case SocialNetwork.POTENTIAL:
                if Model.doProb(self.counts[SocialNetwork.POTENTIAL] * (self.beta_1 * self.counts[SocialNetwork.NEW] + self.beta_2 * self.counts[SocialNetwork.MATURE])):
                    newData["code"] = SocialNetwork.NEW
            case SocialNetwork.NEW:
                if Model.doProb(self.chi * self.counts[SocialNetwork.NEW]):
                    newData["code"] = SocialNetwork.MATURE
                # TODO: check if the probabilities get messed here
                elif Model.doProb(self.delta_1 * self.counts[SocialNetwork.NEW]):
                    newData["code"] = SocialNetwork.MATURE
            case SocialNetwork.MATURE:
                if Model.doProb(self.counts[SocialNetwork.MATURE] * omega):
                    newData["code"] = SocialNetwork.RETIRED
            case SocialNetwork.RETIRED:
                if Model.doProb(self.gamma * self.counts[SocialNetwork.RETIRED]):
                    newData["code"] = SocialNetwork.POTENTIAL
        return {id: newData}


# Wrapper class for array of new states for nodes in model
class ModelUpdate:
    def __init__(self, changes):
        # TODO; fix
        self.changes = dict()
        for change in changes:
            self.changes.update(change)

    def dump(self):
        return self.changes


network = SocialNetwork(200, 100)
for _ in range(200):
    network.draw()
    print("Step", network.model.step_num)
    network.step()
