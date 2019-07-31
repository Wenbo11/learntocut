from gurobienv import *
import os

num = 50
A0 = np.array([[1,-num],[1,num]], dtype=np.float64)
b0 = np.array([0,num], dtype=np.float64)
c0 = -np.array([10,0], dtype=np.float64)
#A = np.abs(np.random.randn(20,1000)) + 0.1
#b = np.zeros([20]) + 1.0
#c = -np.abs(np.random.randn(1000))
env = GurobiEnv()
tdict = []
for _ in range(100):
    if True:
        _,_,a,b,done,_,_,_ = env.reset(A0.copy(),b0.copy(),c0.copy())
        t = 0
        while not done:
            idx = np.random.randint(0,b.size,size=1)[0]
            a = a[idx]
            b = b[idx]
            _,_,a,b,done,obj,_,_ = env.step(a,b)
            t += 1   
        tdict.append(t)
        print('final',obj)
        #os.remove('gurobilog')
        #del env
    else:
    	print('error')
print(np.min(tdict),np.max(tdict),np.mean(tdict))