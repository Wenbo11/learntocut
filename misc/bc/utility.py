from Queue import PriorityQueue, Queue, LifoQueue
import numpy as np

class Node(object):
    def __init__(self, A, b, c, solution=None):
        self.A = A
        self.b = b
        self.c = c
        self.solution = solution

class NodeList(object):
	def __init__(self):
		self.nodes = []
		self.priorities = []

	def append(self, node, priority=1.0):
		self.nodes.append(node)
		self.priorities.append(priority)

	def sample(self):
		# choose the node with highest priority
		idx = np.argmax(self.priorities)
		node = self.nodes.pop(idx)
		self.priorities.pop(idx)
		return node

	def __len__(self):
		return len(self.nodes)

class NodeFIFOQueue(object):
	def __init__(self):
		self.nodes = Queue()
		self.priorities = []

	def append(self, node, prioriy=1.0):
		self.nodes.put(node)
		self.priorities.append(prioriy)

	def sample(self):
		return self.nodes.get()

	def __len__(self):
		return self.nodes.qsize()

class NodeLIFOQueue(object):
	def __init__(self):
		self.nodes = LifoQueue()
		self.priorities = []

	def append(self, node, prioriy=1.0):
		self.nodes.put(node)
		self.priorities.append(prioriy)

	def sample(self):
		return self.nodes.get()

	def __len__(self):
		return self.nodes.qsize()


def checkintegral(x):
	x = np.array(x)
	if np.sum(abs(np.round(x) - x) > 1e-2) >= 1:
		return False
	else:
		return True
