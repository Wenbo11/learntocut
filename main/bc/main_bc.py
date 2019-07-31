import numpy as np
import sys
import time

from misc.bc.solverutils import  GurobiIntSolve, GurobiIntSolve2
from misc.bc.utility import Node, NodeList, NodeFIFOQueue, NodeLIFOQueue, checkintegral
from misc.bc.expandnode_utility import LPExpander,BaselineCutExpander
from misc.bc.make_policy import load_policy

# Here we input the true objective of LP and IP to calculate the IGC
objslp = None
objsip = None
assert objsip is not None and objslp is not None
initial_lp_objective = objslp

# hyperparameters
RATIO_THRESHOLD = 0.0001 # termination condition for BB
TIMELIMIT = 1000 # termination time step
max_num_cuts = 10 # max cuts added to each node
backtrack = False # do backtrack
window = 5 # backtrack window
threshold = 0.01 # backtrack threshold 
baselinemode = str(sys.argv[2]) 
policyNAME = '' # directory to load policy

policy = load_policy(seed=0, n_directions=10, step_size=0.01, delta_std=0.02, policy_type='attention', numvars=30, logdir=policyNAME)

load_dir = '' # IP instance directory
idxinstance = np.int(sys.argv[1]) # which instance to load
print('loading training instances')
Aorig = np.load('{}/A_{}.npy'.format(load_dir, idxinstance))
borig = np.load('{}/b_{}.npy'.format(load_dir, idxinstance))
corig = np.load('{}/c_{}.npy'.format(load_dir, idxinstance))
IPsolution = np.load('{}/solution.npy'.format(load_dir))[idxinstance]

tstart = time.time()
tend = time.time()
time.sleep(.1)
tgurobi = tend - tstart
tstart = time.time()

count = 0
countidx = idxinstance
A = np.load('../backtrack_cuts/A_{}.npy'.format(countidx))
b = np.load('../backtrack_cuts/b_{}.npy'.format(countidx))
time.sleep(.1)

# node expander
if baselinemode == 'None':
	expander = LPExpander()
else:
	expander = BaselineCutExpander(max_num_cuts=max_num_cuts, backtrack=backtrack, mode=baselinemode, policy=policy, window=window, threshold=threshold)

# create an initial node
node = Node(Aorig, borig, corig, IPsolution)

#nodelist = NodeList()
nodelist = NodeFIFOQueue()
#nodelist = NodeLIFOQueue()

# create a list to keep track of fractional solution
# to form the lower bound on the objective
fractionalsolutions = []
childrennodes = []
expanded = []

# create initial best obj and solution
BestObjective = np.inf
BestSolution = None

nodelist.append(node)

# book keepinng
timecount = 0
ratios = []
optimalitygap = []

# main loop
while len(nodelist) >= 1:

	# pop a node
	node = nodelist.sample()

	# load and expand a node
	#feasible, objective, solution = GurobiIntSolve2(A, b, c)
	originalnumcuts = node.A.shape[0]
	feasible, objective, solution, cutfeasible = expander.expandnode(node)
	A, b, c = node.A, node.b, node.c
	newnumcuts = node.A.shape[0]
	print('adding num of cuts {}'.format(newnumcuts - originalnumcuts))

	if feasible:
		assert objective is not None

	# check if thte popped node is the child node of some parent node
	for idx in range(len(childrennodes)):
		if childrennodes[idx][0] == node:
			expanded[idx][0] = 1
			if expanded[idx][1] == 1:
				# pop the corresponding child node
				childrennodes.pop(idx)
				expanded.pop(idx)
				fractionalsolutions.pop(idx)
			break
		elif childrennodes[idx][1] == node:
			expanded[idx][1] = 1
			if expanded[idx][0] == 1:
				# pop the corresponding child node
				childrennodes.pop(idx)
				expanded.pop(idx)
				fractionalsolutions.pop(idx)
			break

	# check cases
	if feasible and objective > BestObjective:
		# prune the node
		pass 
	elif not feasible:
		# prune the node
		pass
	elif checkintegral(solution) is False:
		# the solution is not integer
		# need to branch

		# now we choose branching randomly
		# we choose branching based on how fraction variables are
		index = np.argmax(np.abs(np.round(solution) - solution))
		print(index)

		# add the corresponding constraints and create nodes
		lower_constraint = np.zeros(A.shape[1])
		lower_constraint[index] = 1.0
 		lower = np.floor(solution[index])
 		Alower = np.vstack((A, lower_constraint))
 		blower = np.append(b, lower)
 		node1 = Node(Alower, blower, c, IPsolution)

 		upper_constraint = np.zeros(A.shape[1])
 		upper_constraint[index] = -1.0
		upper = -np.ceil(solution[index])
 		Aupper = np.vstack((A, upper_constraint))
 		bupper = np.append(b, upper)
 		node2 = Node(Aupper, bupper, c, IPsolution)

 		# add nodes to the queue
 		nodelist.append(node1)
 		nodelist.append(node2)

 		# record the newly added child nodes and the fractional solution
 		fractionalsolutions.append(objective)
 		childrennodes.append([node1, node2])
 		expanded.append([0, 0])

 	elif checkintegral(solution) is True:
 		# check if better than current best
 		if objective <= BestObjective:
 			BestSolution = solution
 			BestObjective = objective
 	else:
 		raise NotImplementedError

 	if len(fractionalsolutions) == 0:
 		break

 	print('obj', BestObjective, 'sol', BestSolution, 'num of remaining nodes', len(nodelist), 'check int', checkintegral(solution), 'feasible', feasible)
 	print('lower bound', np.min(fractionalsolutions), 'len of fractional solutions', len(fractionalsolutions))
 	print('lower bound set', fractionalsolutions)
	print('cut is feasible?', cutfeasible)

	# compute optimality gap (old way)
	ratiogap = (np.min(fractionalsolutions) - objslp) / (objsip[idxinstance] - objslp)
	print('objective ratio gap', ratiogap)
	optimalitygap.append(ratiogap)

	# increment time count
	timecount += 1
	if BestSolution is not None:
		# compute the ratio
		gap_now = BestObjective - np.min(fractionalsolutions)
		base_gap = BestObjective - initial_lp_objective
		ratio = gap_now / base_gap
		print('success statistics', ratio)
		ratios.append(ratio)
		if ratio <= RATIO_THRESHOLD:
			break

 	time.sleep(.2)

 	if timecount >= TIMELIMIT:
 		break
 