from envs.gurobienv import timelimit_wrapper, GurobiOriginalEnv
from envs.originalspaceenv import OriginalSpace_Env
import numpy as np

def make_gurobi_env(load_dir, indices, timelimit):

	envdict_total = []
	for idx in indices:
		print('loading training instances')
		A = np.load('{}/A_{}.npy'.format(load_dir, idx))
		b = np.load('{}/b_{}.npy'.format(load_dir, idx))
		c = np.load('{}/c_{}.npy'.format(load_dir, idx))
		solution = np.load('{}/solution.npy'.format(load_dir, idx))[idx]
		solution = np.array(solution)
		#assert solution.size == A.shape[1]
		#env = timelimit_wrapper(GurobiOriginalEnv(A,b,-c,solution), timelimit)
		env = timelimit_wrapper(GurobiOriginalEnv(A,b,c,solution), timelimit)
		envdict_total.append(env)

	stats = {'numvars':A.shape[1]}

	return envdict_total, stats
