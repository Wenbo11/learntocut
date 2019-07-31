import numpy as np
try:
    from gurobilpsolver import GurobiSolver
except:
    from .gurobilpsolver import GurobiSolver
try:
    from solverutils import generatecutzeroth, roundmarrays
except:
    from .solverutils import generatecutzeroth, roundmarrays

GRB_OPTIMAL = 2

def checkintegral(x):
    if np.sum(abs(np.round(x) - x) > 1e-2) >= 1:
        return False
    else:
        return True


def getbasicsolution(xfull, basis):
	"""return vector of basic solution"""
	return xfull[basis]


class GurobiEnv(object):

    def __init__(self):
        pass

    def reset(self, A, b, c):
        """initial inequalities defining the system"""

        # set up
        self.A = A.copy()
        self.b = b.copy()
        self.c = c.copy()

        # initialize gurobi model
        self.solver = GurobiSolver(self.c.size)
        self.solver.reset(self.c)
        index = np.arange(self.A.shape[1])
        for i in range(self.A.shape[0]):
        	self.solver.add_cuts(index, self.A[i,:], np.array([self.b[i]]))

        #self.save_model()

        # solve the system
        self.solver.optimize()

        # output tableau
        tab = self.solver.get_tableau()
        x_original = self.solver.get_original_solution().copy()
        x_slack = self.solver.get_slack_solution().copy()
        x_full = np.append(x_original, x_slack)
        objval, status = self.solver.get_objval()
        basis = self.solver.get_basis()

        # get basis and construct full tab
        #print(np.min(basis),np.max(basis))
        x_basis = getbasicsolution(x_full, basis)
        tab = np.column_stack([x_basis, tab])
        #print(tab.shape, x_basis.shape)
        self.x_basis = x_basis
        self.tab = tab

        # round tab
        tab = roundmarrays(tab)

        # check integrality of solution
        if status == GRB_OPTIMAL:
            done = checkintegral(x_basis)
        else:
            print('status code {}'.format(status))
            done = True

        # generate cuts
        cuts_a = []
        cuts_b = []
        for i in range(x_basis.size):
            if abs(round(x_basis[i])-x_basis[i])>1e-2:
                cut_a,cut_b = generatecutzeroth(tab[i,:])
                assert cut_a.size == self.A.shape[0] + self.A.shape[1]
                a = cut_a[0:self.A.shape[1]]
                e = cut_a[self.A.shape[1]:]
                newA = np.dot(self.A.T,e) - a
                newb = np.dot(e,self.b) - cut_b
                cuts_a.append(newA)
                cuts_b.append(newb)
        cuts_a,cuts_b = np.array(cuts_a),np.array(cuts_b)
        #print('cuts',cuts_a,cuts_b,done,x_original,x_basis,basis,objval)
        #print('full',x_full)
        return self.A, self.b, cuts_a, cuts_b, done, objval, x_original, tab

    def step(self, a, b):
        if isinstance(b, float):
            b = np.array([b])
        # add cuts to self.A, self.b for record
        self.A = np.vstack([self.A, a])
        self.b = np.append(self.b, b)

        # add cuts to solver
        index = np.arange(self.A.shape[1])
        self.solver.add_cuts(index, a, b)

        # solve the system
        self.solver.optimize()

        # output tableau
        tab = self.solver.get_tableau()
        x_original = self.solver.get_original_solution().copy()
        x_slack = self.solver.get_slack_solution().copy()
        x_full = np.append(x_original, x_slack)
        objval, status = self.solver.get_objval()
        basis = self.solver.get_basis()

        # get basis and construct full tab
        #print(np.min(basis),np.max(basis))
        x_basis = getbasicsolution(x_full, basis)
        tab = np.column_stack([x_basis, tab])
        self.x_basis = x_basis
        self.tab = tab
         
        # round tab
        tab = roundmarrays(tab)

        # check integrality of solution
        if status == GRB_OPTIMAL:
            done = checkintegral(x_basis)
        else:
            print('status code {}'.format(status))
            done = True

        # generate cuts
        cuts_a = []
        cuts_b = []
        for i in range(x_basis.size):
            if abs(round(x_basis[i])-x_basis[i])>1e-2:
                cut_a,cut_b = generatecutzeroth(tab[i,:])
                assert cut_a.size == self.A.shape[0] + self.A.shape[1]
                a = cut_a[:self.A.shape[1]]
                e = cut_a[self.A.shape[1]:]
                newA = np.dot(self.A.T,e) - a
                newb = np.dot(e,self.b) - cut_b
                cuts_a.append(newA)
                cuts_b.append(newb)
        cuts_a,cuts_b = np.array(cuts_a),np.array(cuts_b)
        #print('cuts',cuts_a,cuts_b,done,x_original,x_basis,basis,objval)
        #print('full',x_full)
        return self.A, self.b, cuts_a, cuts_b, done, objval, x_original, tab    	

    def save_model(self):
    	self.solver.write_model()


"""
num = 5
A = np.array([[1,-num],[1,num]], dtype=np.float64)
b = np.array([0,num], dtype=np.float64)
c = -np.array([10,0], dtype=np.float64)
#A = np.abs(np.random.randn(20,1000)) + 0.1
#b = np.zeros([20]) + 1.0
#c = -np.abs(np.random.randn(1000))
env = GurobiEnv()
_,_,a,b,done,_,_,_ = env.reset(A,b,c)
done = False
t = 0
import time
start = time.time()
while not done:
    idx = np.random.randint(0,b.size,size=1)[0]
    a = a[idx]
    b = b[idx]
    _,_,a,b,done,_,_,_ = env.step(a,b)
    t += 1
end = time.time
print(t,(end-start)/t)
"""
"""
num = 40
A = np.array([[1,-num],[1,num]], dtype=np.float64)
b = np.array([0,num], dtype=np.float64)
c = -np.array([10,0], dtype=np.float64)
#A = np.abs(np.random.randn(20,1000)) + 0.1
#b = np.zeros([20]) + 1.0
#c = -np.abs(np.random.randn(1000))
env = GurobiEnv()
_,_,cuts_a,cuts_b,done,_,_,_ = env.reset(A,b,c)
done = False
t = 0
import time
start = time.time()
while not done:
    idx = np.random.randint(0,cuts_b.size,size=1)[0]
    cut_a = cuts_a[idx]
    cut_b = cuts_b[idx]
    # update A,b
    A = np.vstack((A,cut_a))
    b = np.append(b,cut_b)
    _,_,cuts_a,cuts_b,done,_,_,_ = env.reset(A,b,c)
    t += 1
end = time.time()
print(t,(end-start)/t)
"""


