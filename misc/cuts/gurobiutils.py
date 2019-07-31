import numpy as np
from gurobipy import *

def GurobiIntSolve(A,b,c):
	c = -c # Gurobi default is maximization
	varrange = range(c.size)
	crange = range(b.size)
	m = Model('LP')
	m.params.OutputFlag = 0 #suppres output
	X = m.addVars(varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.INTEGER,
                 obj=c,
                 name="X")
	C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)==b[i] for i in crange),'C')
	m.params.Method = -1 # primal simplex Method = 0
	m.optimize()
	# obtain results
	solution = []; 
	for i in X:
		solution.append(X[i].X);
	solution = np.asarray(solution)
	return m.ObjVal,solution

def GurobiSolveDual(A,b,c,basis_index):
	#print 'solve dual...'
	c = -c
	varrange = range(c.size)
	crange = range(b.size)
	m = Model('LP')
	m.params.OutputFlag = 0 #suppres output
	m.params.Method = 1 # dual simplex
	X = m.addVars(varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                 obj=c,
                 name="X")
	C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)==b[i] for i in crange), name='C')
	for i in X:
		#print i
		if i in basis_index:
			X[i].VBasis = 1
		else:
			X[i].VBasis = 0
	m.optimize()
	# obtain results
	solution = []; basis_index = []; RC = []
	for i in X:
		solution.append(X[i].X);
		RC.append(X[i].getAttr('RC'))
		if X[i].getAttr('VBasis') == 0:
			basis_index.append(i)
	solution = np.asarray(solution)
	RC = np.asarray(RC)
	basis_index = np.asarray(basis_index)	
	return m.ObjVal,solution,basis_index,RC

def GurobiSolve(A,b,c,Method=0):
	#print('solving starts')
	c = -c # Gurobi default is maximization
	varrange = range(c.size)
	crange = range(b.size)
	m = Model('LP')
	m.params.OutputFlag = 0 #suppress output
	X = m.addVars(varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                 obj=c,
                 name="X")
	C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)==b[i] for i in crange),'C')
	m.params.Method = Method # primal simplex Method = 0
	#print('start optimizing...')
	m.optimize()
	# obtain results
	solution = []; basis_index = []; RC = []
	for i in X:
		solution.append(X[i].X);
		RC.append(X[i].getAttr('RC'))
		if X[i].getAttr('VBasis') == 0:
			basis_index.append(i)
	solution = np.asarray(solution)
	RC = np.asarray(RC)
	basis_index = np.asarray(basis_index)
	#print('solving completes')
	return m.ObjVal,solution,basis_index,RC

# == introduce states into the solver to avoid the overhead of
# re-initializing the model every time step
class GurobiSolver(object):
	def __init__(self, A, b, c, Method=0):
		self.A = A
		self.b = b
		self.c = c
		self.Method = Method

	def add_and_solve(self, a, b):
		# add additional constraints and solve
		self.C = self.model.addConstrs((sum(a[i] * self.X[i] for i in self.varrange) <= b), 'cnew_' + str(self.num_new_cuts))
		self.num_new_cuts += 1
		# re-solve
		self.model.optimize()
		# obtain results
		solution = []; basis_index = []; RC = []
		for i in self.X:
			solution.append(self.X[i].X);
			RC.append(self.X[i].getAttr('RC'))
			if self.X[i].getAttr('VBasis') == 0:
				basis_index.append(i)
		solution = np.asarray(solution)
		RC = np.asarray(RC)
		basis_index = np.asarray(basis_index)
		#print('solving completes')
		return m.ObjVal,solution,basis_index,RC

	def init(self, A, b, c, Method=0):				
		# solve for the first time
		c = -c # Gurobi default is maximization
		self.varrange = range(c.size)
		self.crange = range(b.size)
		self.model = m = Model('LP')
		m.params.OutputFlag = 0 #suppress output
		self.X = m.addVars(self.varrange, lb=0.0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
	                 obj=c,
	                 name="X")
		self.C = m.addConstrs((sum(A[i,j]*self.X[j] for j in self.varrange)==b[i] for i in self.crange),'C')
		m.params.Method = Method # primal simplex Method = 0
		#print('start optimizing...')
		m.optimize()
		# obtain results
		solution = []; basis_index = []; RC = []
		for i in self.X:
			solution.append(self.X[i].X);
			RC.append(self.X[i].getAttr('RC'))
			if self.X[i].getAttr('VBasis') == 0:
				basis_index.append(i)
		solution = np.asarray(solution)
		RC = np.asarray(RC)
		basis_index = np.asarray(basis_index)
		self.num_new_cuts = 0
		#print('solving completes')
		return m.ObjVal,solution,basis_index,RC	


def computeoptimaltab(A,b,RC,obj,basis_index):
	'''
	A - A matrix, b - constraint, RC - reduced cost, basis_index - basis 
	'''
	m,n = A.shape
	assert m == b.size; assert n == RC.size
	B = A[:,basis_index]
	try:
		INV = np.linalg.inv(B)
	except:
		print('basisindex length:', basis_index.size)
		print('Ashape:', A.shape)
		raise ValueError
	x = np.dot(INV,b)
	A_ = np.dot(INV,A)
	firstrow = np.append(-obj,RC)
	secondrow = np.column_stack((x,A_))
	tab = np.vstack((firstrow,secondrow))
	return tab

def GurobiSolvetab(tab,c):
	# extract data from the tab
	m,n = tab.shape
	#print c.size,n
	try:
		assert c.size == n-1
	except:
		print(c.size,tab.shape)
		raise ValueError
	A = tab[1:m,1:n]; b = tab[1:n,0]
	obj,sol,basis,rc = GurobiSolve(A,b,c,0) # dual simplex 1
	return obj,sol,basis,rc

def GurobiSolvetabDual(tab,c,basis_index):
	#print 'solve dual tab..'
	# extract data from the tab
	m,n = tab.shape
	#print c.size,n
	try:
		assert c.size == n-1
	except:
		print(c.size,tab.shape)
		raise ValueError
	A = tab[1:m,1:n]; b = tab[1:n,0]
	#print basis_index
	#print b,c
	obj,sol,basis,rc = GurobiSolveDual(A,b,c,basis_index) # dual simplex 1
	return obj,sol,basis,rc

def generatecutzeroth(row):
	###
	# generate cut that includes cost/obj row as well
	###
	n = row.size
	a = row[1:n]
	b = row[0]
	cut_a = a - np.floor(a)
	cut_b = b - np.floor(b)
	return cut_a,cut_b

def generatecut_MIP(row,I,basis_index):
	'''
	generate cut for MIP
	I: set of vars required to be integers
	'''
	n = row.size
	b = row[0]
	a = row[1:n]
	f = a - np.floor(a)
	f0 = b - np.floor(b)
	cut_a = np.zeros(n-1)
	cut_b = 0
	for i in range(n-1):
		if i not in basis_index:
			if i in I:
				if f[i]<=f0:
					cut_a[i] = f[i]/(f0+0.0)
				else:
					cut_a[i] = (1-f[i])/(1+0.0-f0)
			else:
				if a[i]>=0:
					cut_a[i] = a[i]/(f0+0.0)
				else:
					cut_a[i] = -a[i]/(1+0.0-f0)
	cut_b = 1
	return cut_a,cut_b	

def updatetab(tab,cut_a,cut_b,basis_index):
	cut_a = -cut_a
	cut_b = -cut_b
	m,n = tab.shape
	A_ = tab[1:m,1:n]; b_ = tab[1:m,0]; c_ = tab[0,1:n]; obj = tab[0,0]
	Anew1 = np.column_stack((A_,np.zeros(m-1)))
	Anew2 = np.append(cut_a,1)
	Anew = np.vstack((Anew1,Anew2))
	bnew = np.append(b_,cut_b)
	cnew = np.append(c_,0)
	M1 = np.append(obj,cnew)
	M2 = np.column_stack((bnew,Anew))
	newtab = np.vstack((M1,M2))
	basis_index = np.append(basis_index,n-1)
	return newtab,basis_index,Anew,bnew

def PRUNEtab(tab,basis_index,numvar):
	'''
	prune and return a basis_index cleared of redundant slacks
	'''
	aa = np.asarray(basis_index)
	while np.sum(aa>=numvar)>=1:
		tab,basis_index = prunetab(tab,basis_index,numvar)
		aa = np.asarray(basis_index)
	return tab,basis_index

def prunetab(tab,basis_index,numvar):
	'''
	m,n original size of the tab, m: original num of constraints, n: original num of vars (not including slack vars)
	drop the slack variables that enter basis
	'''
	M,N = tab.shape 
	for i in basis_index:
		if i>=numvar:
            # found a slack variable that enters the basis
            # drop the column
			lset = np.where(abs(tab[1:M,i+1]-1)<1e-8)
			l = lset[0][0]
			tab = np.delete(tab,i+1,1)
			tab = np.delete(tab,l+1,0)
			basis_index = list(basis_index)
			basis_index.remove(i)
			for j in range(len(basis_index)):
				if basis_index[j]>i:
					basis_index[j] -= 1
			basis_index = np.asarray(basis_index)
           # print 'pruning...'
			return tab,basis_index
	return tab,basis_index

'''
A = np.asarray([[1,2,2],[3,2,1]])
b = np.asarray([10,9])
c = np.asarray([-10,1,12])

'''

'''
obj,sol,basis,rc = GurobiSolve(A,b,c,0)

print obj
print sol
print basis
print rc

tab = computeoptimaltab(A,b,rc,obj,basis)
print tab
action = 0
cut_a,cut_b = generatecutzeroth(tab[action,:])
newtab,newbasis = updatetab(tab,cut_a,cut_b,basis)
newobj,newsol,newbasis,newrc = GurobiSolvetab(newtab,np.append(c,0))
'''

'''
varrange = range(c.size)
crange = range(b.size)

m = Model("LP")

X = m.addVars(varrange, vtype=GRB.CONTINUOUS,
                 obj=c,
                 name="X")

C = m.addConstrs((sum(A[i,j]*X[j] for j in varrange)==b[i] for i in crange),'C')

m.params.Method = 0
m.params.OutputFlag = 0
m.optimize()

print m.objVal
basis_index = []
for i in X:
	print X[i].getAttr('RC'),X[i].getAttr('VBasis'),X[i].lb,X[i].ub,X[i].X
	if X[i].getAttr('VBasis') == 0:
		basis_index.append(i)
print basis_index
'''
