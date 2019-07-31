# -*- coding: utf-8 -*-
import numpy as np
import time

from misc.cuts.solverutils import SolveLP,computeoptimaltab,generatecutzeroth,updatetab,SolveLPtabDual,PRUNEtab
from misc.cuts.utilsRLIP import checkterminal,computereward

SOLVER = 'GUROBI'
if SOLVER == 'GUROBI':
	from misc.cuts.gurobiutils import GurobiSolve
if SOLVER == 'SCIPY':
	from misc.cuts.scipyutils import ScipyLinProgSolve

def compute_state(A,b,c):
	m,n = A.shape
	assert m == b.size and n == c.size
	A_tilde = np.column_stack((A,np.eye(m)))
	b_tilde = b
	c_tilde = np.append(c,np.zeros(m))
	if SOLVER == 'HAND':
		#print('not using Gurobi')
		obj,sol,basis_index,rc = SolveLP(A_tilde,b_tilde,c_tilde)
	elif SOLVER == 'GUROBI':
		#print('sovling using Gurobi')
		obj,sol,basis_index,rc = GurobiSolve(A_tilde,b_tilde,c_tilde)
	elif SOLVER == 'SCIPY':
		obj,sol,basis_index,rc = ScipyLinProgSolve(A_tilde,b_tilde,c_tilde)
	tab = computeoptimaltab(A_tilde,b_tilde,rc,obj,basis_index)
	tab = roundmarrays(tab)
	x = tab[:,0]
	#print tab
	done = True
	if np.sum(abs(np.round(x)-x)>1e-2) >= 1:
		done = False
	cuts_a = []
	cuts_b = []
	for i in range(x.size):
		if abs(round(x[i])-x[i])>1e-2: 
			# fractional rows used to compute cut
			cut_a,cut_b = generatecutzeroth(tab[i,:])	
			# a^T x + e^T y >= d
			assert cut_a.size == m+n
			a = cut_a[0:n]
			e = cut_a[n:]
			newA = np.dot(A.T,e) - a
			newb = np.dot(e,b) - cut_b
			cuts_a.append(newA)
			cuts_b.append(newb)
	cuts_a,cuts_b = np.array(cuts_a),np.array(cuts_b)
	return A,b,cuts_a,cuts_b,done,obj,x,tab


def roundmarrays(x,delta=1e-7):
	'''
	if certain components of x are very close to integers, round them
	'''
	index = np.where(abs(np.round(x)-x)<delta)
	x[index] = np.round(x)[index]
	return x

def LP_recursive_drop(A,b):
	excludeList = []
	for i in reversed(range(A.shape[0])):
		try:
			ind = np.ones((A.shape[0],),bool)
			ind[i] = False
			a_examined = A[i,:]
			b_examined =b[i]
			A_others = A[ind,:]
			b_others = b[ind]
			#print(A_others,b_others,a_examined)
			_,_,_,_,_,obj = compute_state(A_others,b_others,a_examined)
			obj = -obj
			if obj <= b_examined+1e-5:
				# should drop this constraint
				excludeList.append(i)
				totallist = range(b.size)
				leftlist = list(set(totallist)-set(excludeList))
				A = A[np.array(leftlist),:]
				b = b[np.array(leftlist)]
				return LP_recursive_drop(A,b)
		except:
			pass
	return A,b


class OriginalSpace_Env(object):
	def __init__(self,A,b,c):
		'''
		max c^T x, Ax <= b, x>=0
		'''
		self.A0 = A.copy()
		self.A = A.copy()
		self.b0 = b.copy()
		self.b = b.copy()
		self.c0 = c.copy()
		self.c = c.copy()
		self.x = None  # current solution

	def reset(self):
		#print('resetting...')
		#print(self.A0.shape, self.b0.shape, self.c0.shape)
		self.A,self.b,self.cuts_a,self.cuts_b,self.done,self.oldobj,self.x,self.tab = compute_state(self.A0,self.b0,self.c0)
		#print('compute state completes...')
		return (self.A,self.b,self.c0,self.cuts_a,self.cuts_b),self.done

	def step(self,action):
		'''
		action is ith row of the cuts matrix
		'''
		cut_a,cut_b = self.cuts_a[action,:],self.cuts_b[action]
		self.A = np.vstack((self.A,cut_a))
		self.b = np.append(self.b,cut_b)
		#self.A,self.b,self.cuts_a,self.cuts_b = map(roundmarrays,[self.A,self.b,self.cuts_a,self.cuts_b])
		#if np.random.rand() < 0.1:
		#	self.LP_drop()  # critical: drop redundant rows
		#self.fast_drop()
		try:
			self.A,self.b,self.cuts_a,self.cuts_b,self.done,self.newobj,self.x,self.tab = compute_state(self.A,self.b,self.c0)
			reward = self.compute_reward()	
		except:
			print('error in lp iteration')
			self.done = True
			reward = -10.0
		objimp = np.clip(abs(self.oldobj - self.newobj),0,1.)
		self.oldobj = self.newobj
		self.A,self.b,self.cuts_a,self.cuts_b = map(roundmarrays,[self.A,self.b,self.cuts_a,self.cuts_b])
		return 	(self.A,self.b,self.c0,self.cuts_a,self.cuts_b),reward,self.done

	def naive_drop(self):
		D = np.zeros((self.b.size,self.b.size))
		for i in range(self.b.size):
			for j in range(self.b.size):
				D[i,j] = D[j,i] = np.sum((self.A[i,:]-self.A[j,:])**2) + (self.b[i]-self.b[j])**2
		excludeList = []
		for i in range(self.b.size-1):
			for j in range(1+i,self.b.size):
				if D[i,j] < 1e-3:
					excludeList.append(j)
		excludeList = list(set(excludeList))
		totallist = range(self.b.size)
		leftlist = list(set(totallist)-set(excludeList))
		self.A = self.A[np.array(leftlist),:]
		self.b = self.b[np.array(leftlist)]
		#print 'dropping %d constraints, leaving %d constraints' % (len(excludeList),self.b.size)

	def LP_drop(self):
		# check if a^T x <= b is redundant in Ax<=b
		# for all (a,b) in (A,b)
		oldbsize = self.b.size
		self.A,self.b = LP_recursive_drop(self.A,self.b)
		newbsize = self.b.size
		#print 'dropping %d constraints, leaving %d constraints' % (oldbsize-newbsize,newbsize)

	def fast_drop(self):
		# check for redundant rows and drop
		array = np.column_stack((self.A,self.b)).copy()
		array = np.round(array)
		"""
		new_array = np.unique(array, axis=0)
		print('old size',array.shape[0],'new size',new_array.shape[0])
		"""
		new_array = np.vstack({tuple(np.round(row)) for row in array})
		self.A = new_array[:,:-1]
		self.b = new_array[:,-1]

	def compute_reward(self):
		if self.done:
			reward = 0.0 #-0.1
		else:
			reward = -1.0 #1.0 #-0.1#objimp
		return reward


from misc.cuts.gurobiutils import GurobiSolver
class OriginalSpaceEnv_fast(OriginalSpace_Env):
	def __init__(self, *args, **kwargs):
		OriginalSpace_Env.__init__(self, *args, **kwargs)
		self.solver = GurobiSolver(self.A0, self.b0, self.c0, Method=0)

	def reset(self):
		self.A,self.b,self.cuts_a,self.cuts_b,self.done,self.oldobj,self.x,self.tab = self.compute_state_init(self.A0,self.b0,self.c0)
		#print('compute state completes...')
		return (self.A,self.b,self.c0,self.cuts_a,self.cuts_b),self.done	

	def step(self, action):
		cut_a,cut_b = self.cuts_a[action,:],self.cuts_b[action]
		self.A = np.vstack((self.A,cut_a))
		self.b = np.append(self.b,cut_b)
		#self.A,self.b,self.cuts_a,self.cuts_b = map(roundmarrays,[self.A,self.b,self.cuts_a,self.cuts_b])
		#if np.random.rand() < 0.1:
		#	self.LP_drop()  # critical: drop redundant rows
		#self.fast_drop()
		self.A,self.b,self.cuts_a,self.cuts_b,self.done,self.newobj,self.x,self.tab = self.compute_state_step(self.A, self.b, self.c0, cut_a, cut_b)	
		objimp = np.clip(abs(self.oldobj - self.newobj),0,1.)
		reward = self.compute_reward()
		self.oldobj = self.newobj
		self.A,self.b,self.cuts_a,self.cuts_b = map(roundmarrays,[self.A,self.b,self.cuts_a,self.cuts_b])
		return 	(self.A,self.b,self.c0,self.cuts_a,self.cuts_b),reward,self.done

	def compute_state_step(self, A, b, c, e, d):
		m,n = A.shape
		assert m == b.size and n == c.size
		A_tilde = np.column_stack((A,np.eye(m)))
		b_tilde = b
		c_tilde = np.append(c,np.zeros(m))
		obj,sol,basis_index,rc = self.solver.add_and_solve(e, d)
		tab = computeoptimaltab(A_tilde,b_tilde,rc,obj,basis_index)
		tab = roundmarrays(tab)
		x = tab[:,0]
		#print tab
		done = True
		if np.sum(abs(np.round(x)-x)>1e-2) >= 1:
			done = False
		cuts_a = []
		cuts_b = []
		for i in range(x.size):
			if abs(round(x[i])-x[i])>1e-2: 
				# fractional rows used to compute cut
				cut_a,cut_b = generatecutzeroth(tab[i,:])	
				# a^T x + e^T y >= d
				assert cut_a.size == m+n
				a = cut_a[0:n]
				e = cut_a[n:]
				newA = np.dot(A.T,e) - a
				newb = np.dot(e,b) - cut_b
				cuts_a.append(newA)
				cuts_b.append(newb)
		cuts_a,cuts_b = np.array(cuts_a),np.array(cuts_b)
		return A,b,cuts_a,cuts_b,done,obj,x,tab		

	def compute_state_init(self, A, b, c):
		m,n = A.shape
		assert m == b.size and n == c.size
		A_tilde = np.column_stack((A,np.eye(m)))
		b_tilde = b
		c_tilde = np.append(c,np.zeros(m))
		obj,sol,basis_index,rc = self.solver.init(A, b, c)
		tab = computeoptimaltab(A_tilde,b_tilde,rc,obj,basis_index)
		tab = roundmarrays(tab)
		x = tab[:,0]
		#print tab
		done = True
		if np.sum(abs(np.round(x)-x)>1e-2) >= 1:
			done = False
		cuts_a = []
		cuts_b = []
		for i in range(x.size):
			if abs(round(x[i])-x[i])>1e-2: 
				# fractional rows used to compute cut
				cut_a,cut_b = generatecutzeroth(tab[i,:])	
				# a^T x + e^T y >= d
				assert cut_a.size == m+n
				a = cut_a[0:n]
				e = cut_a[n:]
				newA = np.dot(A.T,e) - a
				newb = np.dot(e,b) - cut_b
				cuts_a.append(newA)
				cuts_b.append(newb)
		cuts_a,cuts_b = np.array(cuts_a),np.array(cuts_b)
		return A,b,cuts_a,cuts_b,done,obj,x,tab




class ObjGapEnv(OriginalSpace_Env):
	def __init__(self, *args, **kwargs):
		OriginalSpace_Env.__init__(self, *args, **kwargs)

	def compute_reward(self):
		return abs(self.newobj - self.oldobj)


import collections
class OriginalSpaceFIFO_Env(OriginalSpace_Env):
	def __init__(self, maxrows, *args, **kwargs):
		OriginalSpace_Env.__init__(self, *args, **kwargs)
		self.maxrows = maxrows # set the number of rows
		self.originalnumrows = self.A0.shape[0]
		assert self.originalnumrows <= self.maxrows
		self.cut_collections_a = collections.deque(maxlen=self.maxrows - self.originalnumrows)
		self.cut_collections_b = collections.deque(maxlen=self.maxrows - self.originalnumrows)
		for i in range(self.maxrows - self.originalnumrows):
			self.cut_collections_a.append(np.zeros(self.A0.shape[1]))
			self.cut_collections_b.append(0.0)

	def reset(self):
		#print('resetting...')
		#print(self.A0.shape, self.b0.shape, self.c0.shape)
		self.A,self.b,self.cuts_a,self.cuts_b,self.done,self.oldobj,self.x,self.tab = compute_state(self.A0,self.b0,self.c0)
		#print('compute state completes...')
		# clear the fifo buffer
		for i in range(self.maxrows - self.originalnumrows):
			self.cut_collections_a.append(np.zeros(self.A0.shape[1]))
			self.cut_collections_b.append(0.0)
		return (self.A,self.b,self.c0,self.cuts_a,self.cuts_b),self.done

	def step(self,action):
		'''
		action is ith row of the cuts matrix
		'''
		cut_a,cut_b = self.cuts_a[action,:],self.cuts_b[action]

		# add newly added cuts to the fifo buffer
		# drop old cuts automatically
		self.cut_collections_a.append(cut_a)
		self.cut_collections_b.append(cut_b)

		self.A = np.vstack((self.A0, np.array(self.cut_collections_a)))
		self.b = np.append(self.b0, np.array(self.cut_collections_b))

		#self.A,self.b,self.cuts_a,self.cuts_b = map(roundmarrays,[self.A,self.b,self.cuts_a,self.cuts_b])
		#if np.random.rand() < 0.1:
		#	self.LP_drop()  # critical: drop redundant rows
		#self.fast_drop()
		self.A,self.b,self.cuts_a,self.cuts_b,self.done,self.newobj,self.x,self.tab = compute_state(self.A,self.b,self.c0)	
		objimp = np.clip(abs(self.oldobj - self.newobj),0,1.)
		reward = self.compute_reward()
		self.oldobj = self.newobj
		self.A,self.b,self.cuts_a,self.cuts_b = map(roundmarrays,[self.A,self.b,self.cuts_a,self.cuts_b])
		return 	(self.A,self.b,self.c0,self.cuts_a,self.cuts_b),reward,self.done



class timelimit_wrapper(object):

	def __init__(self, env, timelimit):
		self.env = env
		self.timelimit = timelimit
		self.counter = 0

	def reset(self):
		self.counter = 0
		return self.env.reset()

	def step(self, action):
		#print('stepping, counter {}'.format(self.counter))
		self.counter += 1
		obs, reward, done = self.env.step(action)
		if self.counter >= self.timelimit:
			done = True
			print('forced return due to timelimit')
		return obs, reward, done


class empty_wrapper(object):
	def __init__(self, env):
		self.env = env

	def reset(self):
		return self.env.reset()

	def step(self, action):
		return self.env.step(action)


def unimodularize(U,A,b,c):
	Au = np.dot(A,U.T)
	bu = b
	cu = np.dot(U,c)
	return Au,bu,cu


def standard_unimodular_system(U,A,b,c,upperbound=1000):
	# add nonnegative constraint
	A = np.vstack((A,np.array([-1,0])))
	b = np.append(b,0)
	Au,bu,cu = unimodularize(U,A,b,c)
	A_final = np.column_stack((Au,-Au))
	b_final = bu
	c_final = np.append(cu,-cu)
	#set a bound for x+ and x-
	nvar = c_final.size
	A_final = np.vstack((A_final,np.eye(nvar)))
	b_final = np.append(b_final,np.ones(nvar)*upperbound)
	return A_final,b_final,c_final
