import numpy as np

from misc.bc.solverutils import GurobiIntSolve2
import envs.gurobienv
from envs.gurobienv import timelimit_wrapper, GurobiOriginalCutBBEnv

class NodeExpander(object):
    def __init__(self):
        pass

    def expandnode(self, node):
        # return expanded result and modify the node
        raise NotImplementedError


class LPExpander(NodeExpander):
    def __init__(self):
        NodeExpander.__init__(self)

    def expandnode(self, node):
        A, b, c = node.A, node.b, node.c
        feasible, objective, solution = GurobiIntSolve2(A, b, c)
        return feasible, objective, solution, True

#TODO: add parent node
class BaselineCutExpander(NodeExpander):
    def __init__(self, max_num_cuts, backtrack=False, mode=None, policy=None, window=None, threshold=None):
        NodeExpander.__init__(self)
        self.cutadder = BaselineCutAdder(max_num_cuts, backtrack, mode, policy, window, threshold)

    def expandnode(self, node):
        A, b, c = node.A, node.b, node.c
        ipsolution = node.solution
        # solve lp to check if the problem is feasible
        lpfeasible, objective, lpsolution = GurobiIntSolve2(A, b, c)
        print('lp feasible', lpfeasible)
        if lpfeasible:
            # we can add cuts
            Anew, bnew, cutfeasible = self.cutadder.add_cuts(A, b, c, ipsolution)
            # solve the new lp
            newlpfeasible, newobjective, newlpsolution = GurobiIntSolve2(Anew, bnew, c) 
            # modify nodes
            node.A = Anew
            node.b = bnew
            return newlpfeasible, newobjective, newlpsolution, cutfeasible
        else:
            return lpfeasible, objective, lpsolution, True

# ====
# class for adding cuts 
# ====

class CutAdder(object):
    def __init__(self):
        pass

    def add_cuts(self):
        raise NotImplementedError

class BaselineCutAdder(CutAdder):
    def __init__(self, max_num_cuts, backtrack, mode, policy, window=None, threshold=None):
        CutAdder.__init__(self)
        self.max_num_cuts = max_num_cuts
        self.backtrack = backtrack
        self.mode = mode
        self.policy = policy
        self.window = window
        self.threshold = threshold
        assert self.mode in ['random','maxviolation','maxnormviolation','rl']
        if self.mode == 'rl':
        	assert policy is not None

    def add_cuts(self, A, b, c, solution):
        env = timelimit_wrapper(GurobiOriginalCutBBEnv(A, b, c,solution), timelimit=self.max_num_cuts)
        A, b, feasible = elementaryrollout(env, self.policy, rollout_length=self.max_num_cuts, gamma=1.0, mode=self.mode, backtrack=self.backtrack, window=self.window, threshold=self.threshold)
        return A, b, feasible


# ====
# common function to add cuts
# ====
def elementaryrollout(env, policy, rollout_length, gamma, mode, backtrack, window=None, threshold=None):
    # take in an environment
    # run cutting plane adding until termination
    # return both the LP bound and two newly branched LPs

    if backtrack:
    	assert window is not None and threshold is not None

    A_orig = env.env.baseenv.A.copy()
    b_orig = env.env.baseenv.b.copy()

    if mode == 'rl':
        assert policy is not None
    rewards = []
    objs = []
    cutoffs = []
    times = []
    if True:
        ob, _ = env.reset()
        factor = 1.0
        #ob = env.reset()
        done = False
        t = 0
        rsum = 0
        cutoff = []
        obj = []
        backtrack_stats = []
        while not done and t <= rollout_length:
            #try:
            if True:
                if mode == 'rl':
                    action = policy.act(ob)
                # random acttion
                elif mode == 'random':
                    _,_,_,cutsa,cutsb = ob
                    if cutsb.size >= 1:
                        action = np.random.randint(0, cutsb.size, size=1)[0]
                    else:
                        action = []
                elif mode == 'maxviolation':
                    x = env.env.baseenv.x_basis.copy()
                    # reduce the solution to fractional part only
                    x_frac = []
                    for i in range(x.size):
                        if abs(x[i] - round(x[i])) > 1e-2:
                            x_frac.append(abs(x[i] - round(x[i])))
                    if len(x_frac) >= 1:
                        action = np.argmax(x_frac)
                    else:
                        action = []
                elif mode == 'maxnormviolation':
                    x = env.env.baseenv.x_basis.copy()
                    tab = env.env.baseenv.tab.copy()
                    # reduce the solution to fractional part only
                    x_frac = []
                    #print(x.shape)
                    for i in range(x.size):
                        if abs(x[i] - round(x[i])) > 1e-2:
                            x_frac.append(abs(x[i] - round(x[i])) / np.linalg.norm(tab[i,1:] + 1e-8))
                    if len(x_frac) >= 1:
                        action = np.argmax(x_frac)
                    else:
                        action = []
                else:
                    raise NotImplementedError
            #except:
            else:
                print('breaking')
                print(env.env.x)
                #print(env.env.done)
                break # this case is when adding one branch terminates the process
            #print(action)
            # random 
            #_,_,_,cutsa,cutsb = ob
            #action = np.random.randint(0, cutsb.size, size=1)[0]
            #ob, r, done = env.step(action)
            ob, r, done  = env.step(action)
            rsum += r * factor
            factor *= gamma
            t += 1
            if r < 0:
                # cut off
                cutoff.append(1)
                gap = r + 1000.0
                obj.append(gap)
            else:
                cutoff.append(0)
                gap = r
                obj.append(r)

            # ==== backtracking mechanism ====
            if backtrack:
                backtrack_stats.append(r / np.sum(obj))
                if len(backtrack_stats) >= window:
                    last_stats = backtrack_stats[-window:]
                    last_stats = np.array(last_stats)
                    if np.sum(last_stats <= threshold) == window:
                        # save the cuts when backtrack stops
                        #np.save('backtrack_cuts/A_{}'.format(COUNT), env.env.baseenv.A)
                        #np.save('backtrack_cuts/b_{}'.format(COUNT), env.env.baseenv.b)
                        #OUNT += 1
                        break
            # ==========
        #np.save('random_backtrack_cuts/A_{}'.format(COUNT), env.env.baseenv.A)
        #np.save('random_backtrack_cuts/b_{}'.format(COUNT), env.env.baseenv.b)
        #COUNT += 1
        objs.append(obj)
        cutoffs.append(cutoff)
        rewards.append(rsum)
        times.append(t)

    A = env.env.baseenv.A.copy()
    b = env.env.baseenv.b.copy()

    # here we check if the cuts have cut off the optimal solution
    # if the original LP is infeasible - this does not matter
    feasible_original = gurobienv.check_feasibility(A_orig, b_orig, env.env.IPsolution)
    feasible_later = gurobienv.check_feasibility(A, b, env.env.IPsolution)
    feasible_cut = True
    if feasible_original:
        if not feasible_later:
            feasible_cut = False

    return A, b, feasible_cut



















