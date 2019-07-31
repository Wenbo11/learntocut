import numpy as np

def generate_tsp(n):
	# n: num of nodes
	# mtz foromulation introduces
	# n-1 node variables
	# n**2 edge variables
	# n**2+n-1 variables in total
	# edge var x_ij 0<= i,j <= n-1 ---> a[i*n+j]
	# node var u_i 1<= i <= n-1 ---> a[n**2+i-1]
	size_var = n**2 + n - 1
	C = np.reshape(np.random.randint(1, 10, size=n**2),[n,n]) # cost matrix
	
	Adict, bdict = [], []
	# sum_i x_ij = 1
	for i in range(n):
		a = np.zeros(size_var)
		for j in range(n):
			a[i*n+j] = 1
		b = 1
		Adict.append(a)
		bdict.append(b)

		a = np.zeros(size_var)
		for j in range(n):
			a[i*n+j] = -1
		b = -1
		Adict.append(a)
		bdict.append(b)

	# sum_j x_ij = 1
	for j in range(n):
		a = np.zeros(size_var)
		for i in range(n):
			a[i*n+j] = 1
		b = 1
		Adict.append(a)
		bdict.append(b)

		a = np.zeros(size_var)
		for i in range(n):
			a[i*n+j] = -1
		b = -1
		Adict.append(a)
		bdict.append(b)

	# x_ij <= 1
	for i in range(n):
		for j in range(n):
			a = np.zeros(size_var)
			a[i*n+j] = 1
			b = 1
			Adict.append(a)
			bdict.append(b)

	# mtz
	# u_i <= n
	for i in range(1,n):
		a = np.zeros(size_var)
		a[n**2+i-1] = 1
		b = n
		Adict.append(a)
		bdict.append(b)

	# u_i >= 2
	for i in range(1,n):
		a = np.zeros(size_var)
		a[n**2+i-1] = -1
		b = -2
		Adict.append(a)
		bdict.append(b)

	# u_i - u_j +1 \leq (n-1)(1-x_ij)
	for i in range(1,n):
		for j in range(1,n):
			a = np.zeros(size_var)
			a[n**2+i-1] = 1
			a[n**2+j-1] = -1
			a[i*n+j] = n-1
			b = n-2
			Adict.append(a)
			bdict.append(b)

	# cost vector
	c = np.zeros(size_var)
	for i in range(n):
		for j in range(n):
			if i == j:
				c[i*n+j] = 1000
			else:
				c[i*n+j] = C[i,j]

	return np.array(Adict), np.array(bdict), -c	


from gurobipy import *

CUT_NAMES = {
		'Clique:', 'Cover:', 'Flow cover:', 'Flow path:', 'Gomory:', 
		'GUB cover:', 'Inf proof:', 'Implied bound:', 'Lazy constraints:', 
		'Learned:', 'MIR:', 'Mod-K:', 'Network:', 'Projected Implied bound:', 
		'StrongCG:', 'User:', 'Zero half:'}

def GurobiIntSolve(A,b,c):
	#c = -c # Gurobi default is maximization
	varrange = range(c.size)
	crange = range(b.size)
	m = Model('LP')
	#m.params.OutputFlag = 0 #suppres output
	X = m.addVars(varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER,
                 obj=c,
                 name="X")
	m.setObjective(sum(c[j]*X[j] for j in varrange), sense=GRB.MINIMIZE)
	C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)<=b[i] for i in crange),'C')
	m.params.Method = -1 # primal simplex Method = 0
	m.write('vertexcover.lp')

	m.Params.timelimit = 300.0

	# setup counting cuts
	#m._cut_count = {k:0 for k in CUT_NAMES} 
	#m._cut_total_count = 0

	# turn on gurobi passes
	#assert hasattr(m.Params, 'GurobiPasses')
	#m.Params.GurobiPasses = 1000000

	#assert hasattr(m.Params, 'Cuts')
	m.Params.Cuts = 0

	#assert hasattr(m.Params, 'Presolve')
	m.Params.Presolve = 0

	#assert hasattr(m.Params, 'Heuristics')
	m.Params.Heuristics = 0

	m.setParam('Heuristics', 0)
	m.setParam('Presolve', 0)
	m.setParam('Cuts', 0)

	# turn off other cuts
	#turn_off_cuts(m)

	# turn off output flag
	#m.setParam('OutputFlag', False)

	# optimize
	#m.optimize(cut_counter)
	m.optimize()
	#print('final cut',m._cut_count)
	#print('total cut',m._cut_total_count
	# obtain results
	solution = []; 
	for i in X:
		solution.append(X[i].X);
	solution = np.asarray(solution)
	return m.ObjVal,solution


def GurobiIntSolve2(A,b,c):
        #c = -c # Gurobi default is maximization
        varrange = range(c.size)
        crange = range(b.size)
        m = Model('LP')
        #m.params.OutputFlag = 0 #suppres output
        X = m.addVars(varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                 obj=c,
                 name="X")
        m.setObjective(sum(c[j]*X[j] for j in varrange), sense=GRB.MINIMIZE)
        C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)<=b[i] for i in crange),'C')
        m.params.Method = -1 # primal simplex Method = 0

        # setup counting cuts
        m._cut_count = {k:0 for k in CUT_NAMES}
        m._cut_total_count = 0

        # turn on gurobi passes
        #assert hasattr(m.Params, 'GurobiPasses')
        #m.Params.GurobiPasses = 1000000

        #assert hasattr(m.Params, 'Cuts')
        #m.Params.Cuts = 0

        #assert hasattr(m.Params, 'Presolve')
        #m.Params.Presolve = 0

        #assert hasattr(m.Params, 'Heuristics')
        #m.Params.Heuristics = 0

        # turn off other cuts
        #turn_off_cuts(m)

        # turn off output flag
        m.setParam('OutputFlag', False)

        # optimize
        #m.optimize(cut_counter)
        m.optimize()
        #print('final cut',m._cut_count)
        #print('total cut',m._cut_total_count)
        # obtain results
        solution = []
        # always assume bounded
        feasible = True
        try:
            for i in X:
                solution.append(X[i].X);
            solution = np.asarray(solution)
            objval = m.ObjVal
        except:
            feasible = False
            objval = None
        return feasible, objval, solution # invert the sign of the solution

def turn_off_cuts(m):
	ATTR_CUT_NAMES = {
		'Clique', 'Cover', 'FlowCover', 'FlowPath', 
		'GUBCover', 'InfProof', 'Implied', 'Lazy constraints', 
		'Learned', 'MIR', 'ModK', 'Network', 'ProjImplied', 
		'StrongCG', 'User', 'ZeroHalf'}
	assert 'Gomory' not in ATTR_CUT_NAMES
	CUTS = [name + 'Cuts' for name in ATTR_CUT_NAMES]
	for name in CUTS:
		assert hasattr(m.Params, name)
		setattr(m.Params, name, 0)


def GurobiIntSolve3(A,b,c):
        c = -c # Gurobi default is maximization
        varrange = range(c.size)
        crange = range(b.size)
        m = Model('LP')
        #m.params.OutputFlag = 0 #suppres output
        X = m.addVars(varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                 obj=c,
                 name="X")
        C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)<=b[i] for i in crange),'C')
        m.params.Method = -1 # primal simplex Method = 0

        # setup counting cuts
        m._cut_count = {k:0 for k in CUT_NAMES}
        m._cut_total_count = 0

        # turn on gurobi passes
        assert hasattr(m.Params, 'GurobiPasses')
        m.Params.GurobiPasses = 1000000

        assert hasattr(m.Params, 'Cuts')
        m.Params.Cuts = 3

        assert hasattr(m.Params, 'Presolve')
        m.Params.Presolve = 0

        assert hasattr(m.Params, 'Heuristics')
        m.Params.Heuristics = 0

        # turn off other cuts
        turn_off_cuts(m)

        # optimize
        m.optimize(cut_counter)
        print('final cut',m._cut_count)
        print('total cut',m._cut_total_count)
        # obtain results
        solution = [];
        for i in X:
                solution.append(X[i].X);
        solution = np.asarray(solution)
        return m.ObjVal,solution

def cut_counter(model, where):
	if where == GRB.Callback.MESSAGE:
		# Message callback
		msg = model.cbGet(GRB.Callback.MSG_STRING)
		for name in model._cut_count.keys():
			if name in msg:
				model._cut_count[name] += int(msg.split(':')[1])
		if any(name in msg for name in model._cut_count.keys()):
			model._cut_total_count += int(msg.split(':')[1])
