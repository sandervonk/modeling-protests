# https://networkx.org/documentation/stable/tutorial.html
# https://stackoverflow.com/questions/19212979/draw-graph-in-networkx
import matplotlib.pyplot as pyplot
import networkx as nx

import random
import copy
import json
import os
import subprocess
import numpy

config = json.load(open('config.json', 'r'))
MODEL = config["model"]


# 1 - randomly generate a social network
# 2 - combine difficusion and broadcasting model to represent flow of ideas
# 3 - potential protesters, turnout; spread of how protests are forming withing the network
# 4 - repeat 2nd and 3rd step to simulate spread


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

    def __init__(self, size=20, seed_verts=10, intitial_contacts=2, secondary_contacts=1,
                 folder="plot", graph=None, options=None):
        n, n_0, m_r, m_s = size, seed_verts, intitial_contacts, secondary_contacts

        self.folder = folder
        if graph != None:
            self.graph = graph
            self.size = graph.number_of_nodes()
        else:
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
        if options != None:
            self.model = Model(self, options)

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

        fig = pyplot.gcf()
        SocialNetwork.saveFrame(self.folder, fig, self.model.step_num)

    @staticmethod
    def saveFrame(folder, fig, step_num):
        path = f"./runs/{folder}"
        if not os.path.exists(path):
            os.makedirs(path)

        filename = path + f'/{step_num:04d}.jpg'
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
    def __init__(self, network, OPT):
        self.network = network
        self.step_num = 0
        self.graphs = dict()

        num = network.size                          # Size of input network
        self.n = num                                # Total number of population

        self.delta_2 = OPT["delta_2"]               # Withdrawal rate for mature protestors
        self.delta_1 = OPT["delta_1"]               # Withdrawal rate for new protestors
        self.delta_0 = self.delta_2 - self.delta_1  # Used as shorthand only

        self.chi = OPT["chi"]                       # Rate of new protestors turning into mature protestors
        self.gamma = OPT["gamma"]                   # Rate of retired protestors turining into potential protestors

        self.beta_1 = OPT["beta_1"]                 # Attractiveness to become protestors from new protestors
        self.beta_2 = OPT["beta_2"]                 # Attractiveness to become protestors from mature protestors

        self.inform_lone = OPT["inform_lone"]       # Probability of lone uninformed nodes learning about protests (online?)
        self.inform_each = OPT["inform_each"]       # Contribution of neighbor nodes to converting uninformed nodes

        self.forget = OPT["forget"]*100             # Probability of forgetting experiences and becoming uninformed in retirement, corrected for other prob format

        self.seed()

    def seed(self):
        counts = {getattr(SocialNetwork, key, 0): MODEL["seed"][key] for key in MODEL["seed"]}
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
        for code in self.counts:
            self.graphs[code] = self.graphs.get(code, [])
            self.graphs[code].append(self.counts[code])

        omega = self.omega()
        update = ModelUpdate([self.getChangeState(*node, omega) for node in self.network.graph.nodes(data=True)])
        self.step_num += 1
        self.network.runUpdate(update)

    # implement binominal chance as derived by Pete
    def doBinomial(self, n):
        return Model.doProb((100 * (1 - (1 - self.inform_each)**n)) if n != 0 else 100 * self.inform_lone)

    @staticmethod
    def doProb(chance):
        result = random.random() * 100 < abs(chance)
        return result
    # Probability-based state change, returned as the new (can be unchanged) state

    def activeNeighbors(self, id):
        count = 0
        for neighbor_id in self.network.graph.neighbors(id):
            neighbor = self.network.graph.nodes[neighbor_id]
            if neighbor["code"] != SocialNetwork.UNINFORMED:
                count += 1
        return count

    def getChangeState(self, id, data, omega):
        newData = copy.copy(data)
        match data["code"]:
            case SocialNetwork.UNINFORMED:
                if self.doBinomial(self.activeNeighbors(id)):
                    newData["code"] = SocialNetwork.POTENTIAL
            case SocialNetwork.POTENTIAL:
                if Model.doProb(self.counts[SocialNetwork.POTENTIAL] * (self.beta_1 * self.counts[SocialNetwork.NEW] + self.beta_2 * self.counts[SocialNetwork.MATURE])):
                    newData["code"] = SocialNetwork.NEW
            case SocialNetwork.NEW:
                # TODO: check if the probabilities get messed here
                if Model.doProb(self.chi * self.counts[SocialNetwork.NEW]):
                    newData["code"] = SocialNetwork.MATURE
                elif Model.doProb(self.delta_1 * self.counts[SocialNetwork.NEW]):
                    newData["code"] = SocialNetwork.MATURE
            case SocialNetwork.MATURE:
                if Model.doProb(self.counts[SocialNetwork.MATURE] * omega):
                    newData["code"] = SocialNetwork.RETIRED
            case SocialNetwork.RETIRED:
                if Model.doProb(self.gamma * self.counts[SocialNetwork.RETIRED]):
                    newData["code"] = SocialNetwork.POTENTIAL
                elif Model.doProb(self.forget):
                    newData["code"] = SocialNetwork.UNINFORMED
        return {id: newData}


# Wrapper class for array of new states for nodes in model
class ModelUpdate:
    def __init__(self, changes):
        self.changes = dict()
        for change in changes:
            self.changes.update(change)

    def dump(self):
        return self.changes


def drawGraphs(steps, data, path):

    xaxis = numpy.array(range(config["STEPS"]))
    for code in data:
        pyplot.plot(xaxis, numpy.array(data[code]), color=SocialNetwork.colors[code])
        pyplot.text(xaxis[-1] + 6, data[code][-1], code, color=SocialNetwork.colors[code], fontsize=10, weight="bold")
    pyplot.xlabel(f"Steps ({steps} total)")
    pyplot.ylabel("Member count")
    pyplot.title('Members by code over time')
    pyplot.savefig(path, dpi=300)


if not os.path.exists("./out"):
    os.mkdir("./out")

base_graph = SocialNetwork(MODEL["size"]["total"], MODEL["size"]["seed"]).graph
for run in config["runs"]:
    graph = copy.deepcopy(base_graph)
    network = SocialNetwork(graph=graph, options=config["runs"][run], folder=run)
    for _ in range(config["STEPS"]):
        network.draw()

        print(f"\r{run} Step {network.model.step_num + 1}/{config['STEPS']} ", end="")
        network.step()

    with open(f"./out/{run}.json", "w") as file:
        json.dump(network.model.graphs, file)

    drawGraphs(config["STEPS"], network.model.graphs, f"./out/{run}.jpg")

    # Define the command as a list of arguments
    command = [
        "ffmpeg",
        "-framerate", "20",
        "-i", f"./runs/{run}/%04d.jpg",
        "-c:v", "libx264",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt", "yuv420p",
        f"./out/{run}.mp4",
        "-y"
    ]

    # Run the command
    subprocess.run(command, check=True, capture_output=True)
