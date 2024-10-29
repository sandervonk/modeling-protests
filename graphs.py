# https://python.igraph.org/en/stable/tutorials/quickstart.html
# https://networkx.org/documentation/stable/tutorial.html
import networkx as nx
import random
import matplotlib


class Person:
    def __init__(self, id, membership=[], hasInfo=[]):
        self.id = id
        self.memberOf = set(membership)
        self.hasInfo = set(hasInfo)


class SocialNetwork:
    def __init__(self, numMembers=100, connectTo=0.7):
        self.graph = nx.Graph()
        self.fill(numMembers)
        self.connectToOthers(connectTo)

    def fill(self, size):
        self.size = size
        data = [(id, {"id": id}) for id in range(self.size)]
        self.graph.add_nodes_from(data)

    def connectToOthers(self, num):
        size = self.size
        for id in range(size):
            to = id

            while id == to:
                to = random.randrange(size)
            self.graph.add_edge(id, to)

    def draw(self):
        nx.draw(self.graph)


network = SocialNetwork()
network.draw()
