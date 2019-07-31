import numpy as np
from libs.cwrapping.gurobicpy import GurobiEnv
GurobiEnv = GurobiEnv

def make_float64(lists):
	newlists = []
	for e in lists:
		newlists.append(np.float64(e))
	return newlists

def check_feasibility(A, b, solution):
	RHS = np.dot(A, solution)
	#print(RHS - b)
	#print(RHS - (1.0 - 1e-10) * b)
	if np.sum(RHS - (1.0 - 1e-10) * b > 1e-5) >= 1:
		return False
	else:
		return True

class GurobiOriginalEnv(object):
	def __init__(self, A, b, c, solution):
		A, b, c = make_float64([A, b, c])
		self.baseenv = GurobiEnv()
		self.baseenv.reset(A, b, c)
		self.A0 = A.copy()
		self.b0 = b.copy()
		self.c0 = c.copy()
		#assert A.shape[1] == solution.size
		self.IPsolution = solution # convention

	def reset(self):
		print('resetting and checking ip feasibility')
		A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.reset(self.A0, self.b0, self.c0)
		self.cutsa = cutsa
		self.cutsb = cutsb
		self.objval = objval
		self.x = xfull
		self.tab = tab
		return (A,b,self.c0,cutsa,cutsb),done

	def step(self, action):
		if isinstance(action, list):
			#print('num of cuts to add',len(action))
			if len(action) >= 1:
				for a in action:
					cuta = self.cutsa[a,:]
					cutb = self.cutsb[a]
					A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.step(cuta, cutb)
		elif isinstance(action, int):
			cuta = self.cutsa[action,:]
			cutb = self.cutsb[action]
			A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.step(cuta, cutb)
		else:
			raise NotImplementedError			
		# compute reward
		reward = 0.0 + abs(objval - self.objval)

		self.cutsa = cutsa
		self.cutsb = cutsb
		self.objval = objval
		self.done = done
			
		self.x = xfull
		self.tab = tab

		return (A,b,self.c0,cutsa,cutsb),reward,done


class GurobiOriginalBBEnv(object):
	def __init__(self, A, b, c, solution):
		A, b, c = make_float64([A, b, c])
		self.baseenv = GurobiEnv()
		self.baseenv.reset(A, b, c)
		self.A0 = A.copy()
		self.b0 = b.copy()
		self.c0 = c.copy()
		assert A.shape[1] == solution.size
		self.IPsolution = solution # convention

	def reset(self):
		print('resetting and checking ip feasibility')
		A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.reset(self.A0, self.b0, self.c0)
		self.cutsa = cutsa
		self.cutsb = cutsb
		self.objval = objval
		self.x = xfull
		self.tab = tab

		return (A,b,self.c0,cutsa,cutsb),done

	def step(self, action):
		cuta = self.cutsa[action,:]
		cutb = self.cutsb[action]
		A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.step(cuta, cutb)
		# compute reward
		reward = 0.0 + abs(objval - self.objval)

		self.cutsa = cutsa
		self.cutsb = cutsb
		self.objval = objval
			
		self.x = xfull
		self.tab = tab

		return (A,b,self.c0,cutsa,cutsb),reward,done

class GurobiTSPEnv(GurobiOriginalEnv):
	def __init__(self, *args, **kwargs):
		GurobiOriginalEnv.__init__(self, *args, **kwargs)

	def reset(self):
		A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.reset(self.A0, self.b0, self.c0)
		if cutsb.size == 1:
			done = True
			# record
			self.cutsa = []
			self.cutsb = []
		else:
			self.cutsa = cutsa[1:]
			self.cutsb = cutsb[1:]
		self.objval = objval
		return (A,b,self.c0,self.cutsa,self.cutsb),done

	def step(self, action):
		cuta = self.cutsa[action,:]
		cutb = self.cutsb[action]
		A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.step(cuta, cutb)
		# compute reward
		reward = 0.0 + abs(objval - self.objval)

		# here coeff c is not integer, need to take action >= 1
		if cutsb.size == 1:
			done = True
			# record
			self.cutsa = []
			self.cutsb = []
		else:
			# record
			self.cutsa = cutsa[1:]
			self.cutsb = cutsb[1:]

		self.objval = objval

		# cut reward
		if done:
			reward = 1.0
		else:
			reward = -0.1

		return (A,b,self.c0,self.cutsa,self.cutsb),reward,done

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

# === used for BB procedure
class GurobiOriginalCutBBEnv(object):
	def __init__(self, A, b, c, solution):
		A, b, c = make_float64([A, b, c])
		self.baseenv = GurobiEnv()
		self.baseenv.reset(A, b, c)
		self.A0 = A.copy()
		self.b0 = b.copy()
		self.c0 = c.copy()
		assert A.shape[1] == solution.size
		self.IPsolution = solution # convention

	def reset(self):
		print('resetting and checking ip feasibility')
		A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.reset(self.A0, self.b0, self.c0)
		self.cutsa = cutsa
		self.cutsb = cutsb
		self.objval = objval
		self.x = xfull
		self.tab = tab

		return (A,b,self.c0,cutsa,cutsb),done

	def step(self, action):
		if (isinstance(action, list) and len(action) >= 1) or isinstance(action, int):
			cuta = self.cutsa[action,:]
			cutb = self.cutsb[action]
			A,b,cutsa,cutsb,done,objval,xfull,tab = self.baseenv.step(cuta, cutb)

			# compute reward
			reward = 0.0 + abs(objval - self.objval)
			# record
			#print('cut', cuta, cutb)
			self.cutsa = cutsa
			self.cutsb = cutsb
			self.objval = objval
			self.done = done
				
			self.x = xfull
			self.tab = tab

			#print('original var', xfull)
			self.oldpackage = (A,b,self.c0,cutsa,cutsb),reward,done
			return self.oldpackage

		elif isinstance(action, list) and len(action) == 0:
			return None,0.0,True
		else:
			raise NotImplementedError

