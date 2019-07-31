#from scipy.optimize import linprog
from scipysolver import linprog
import numpy as np


def ScipyLinProgSolve(A_tilde,b_tilde,c_tilde):
	# explicitly add x>=0
	m,n = A_tilde.shape
	A = np.column_stack((A_tilde, np.eye(m)))
	b = b_tilde.copy()
	c = -np.append(c_tilde, np.zeros(m))
	res = linprog(c=c, A_eq=A, b_eq=b, method='simplex', bounds=(0, None))
	rc = res.tableau[-1, :n]
	obj = res.fun
	sol = res.x[:n]
	basis_index = list(res.basis[res.basis <= n-1])
	return obj, sol, basis_index, rc

"""
print('scipy')
c = np.array([1, 2, 3, 2])
A = np.array([[1, 2, 1, 1], [2, 1, 1, 1]])
b = np.array([600, 4])
obj, sol, basis_index, rc = ScipyLinProgSolve(A, b, -c)
print(obj,sol,basis_index,rc)
print('gurobi')
from gurobiutils import GurobiSolve
obj,sol,basis_index,rc = GurobiSolve(A,b,c)
print(obj,sol,basis_index,rc)
print('hand')
from solverutils import SolveLP
obj,sol,basis_index,rc = SolveLP(A,b,c)
print(obj,sol,basis_index,rc)
"""

