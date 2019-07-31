# -*- coding: utf-8 -*-
import numpy as np
import sys
import os

def blockPrint():
	sys.stdout = open(os.devnull,'w')

def enablePrint():
	sys.stdout = sys.__stdout__
	
def checkterminal(tab):
	m,n = tab.shape
	x = tab[:,0]
	if np.sum(abs(x-np.round(x))>1e-10)>=1:
		return False
	else:
		return True

def compute_solution_from_tab(tab,basis_index):
	m,n = tab.shape
	lenx = m - 1
	try:
		assert lenx == len(basis_index)
	except:
		print(tab.shape,basis_index.size)
		raise ValueError
	X = np.zeros(n-1)
	for i in range(lenx):
		X[basis_index[i]] = tab[i+1,0]
	return X

def checksolution(x,I):
	for i in I:
		if abs(x[i] - np.round(x[i]))>1e-10: # is not integer
			return False
	return True

def checkterminal_MIP(tab,basis_index,I):
	'''
	I is the set of variables required to be integers
	'''
	x = compute_solution_from_tab(tab,basis_index)
	return checksolution(x,I)

def findcandidate_MIP(tab,basis_index,I):
	m,n = tab.shape
	x = tab[1:m,0]
	candidate = []
	basis_index = list(basis_index)
	for j in range(len(basis_index)):
		if ((basis_index[j] in I) and (abs(x[j]-np.round(x[j]))>1e-10)):
			candidate.append(j+1)
	return candidate

def computereward(tab,newtab):
	obj = -tab[0,0]; newobj = -newtab[0,0]
	if checkterminal(newtab):
		return 1
	else:
		return -0.1#-0.1
		#+0.01*abs(obj-newobj)/(0.1+abs(newobj)+abs(obj))

def generateinstances(nvar,nc,num):
	Adict = np.zeros((num,nc,nvar+nc))
	bdict = np.zeros((num,nc))
	cdict = np.zeros((num,nvar+nc))
	for i in range(num):
		A1 = np.random.randint(size=(nc,nvar),low=1,high=20)
		A2 = np.eye(nc)
		A = np.column_stack((A1,A2))
		b = np.random.randint(size=nc,low=1,high=(nvar+nc)*25)
		c = np.random.randint(size=nvar,low=-20,high=20)
		c = np.append(c,np.zeros(nc))
		Adict[i,:] = A
		bdict[i,:] = b
		cdict[i,:] = c
	return Adict,bdict,cdict

def generateinstances_MIP(nvar,nc,numintegers,num):
	assert numintegers<=(nvar+nc)
	Adict = np.zeros((num,nc,nvar+nc))
	bdict = np.zeros((num,nc))
	cdict = np.zeros((num,nvar+nc))
	Idict = []
	for i in range(num):
		A1 = np.random.randint(size=(nc,nvar),low=1,high=20)
		A2 = np.eye(nc)
		A = np.column_stack((A1,A2))
		b = np.random.randint(size=nc,low=1,high=(nvar+nc)*25)
		c = np.random.randint(size=nvar,low=-20,high=20)
		c = np.append(c,np.zeros(nc))
		Adict[i,:] = A
		bdict[i,:] = b
		cdict[i,:] = c
		Idict.append(range(numintegers))
	return Adict,bdict,cdict,Idict
