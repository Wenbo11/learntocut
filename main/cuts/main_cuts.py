import numpy as np
import os
import psutil
import time

from es.policies import LinearRowFeaturePolicy, MLPRowFeaturePolicy, MLPRowFeatureAttenttionPolicy, MLPRowFeatureAttenttionEmbeddingPolicy, MLPRowFeatureLSTMEmbeddingPolicy
from es.optimizers import Adam
from es.alg_utils import rollout, rollout_envs, rollout_evaluate
from es.utils import compute_stats
from misc.cuts.make_envs import make_gurobi_env

name = 'randomip' # specify task name
LOGDIR = '../../data/randomip' # specify the directory in which to find raw IP instances

# the program maintains a memory usage of less than MEM_THRESHOLD of the total memory of the machine
# this is to partially resolve the memory leak due to the simulation environment
MEM_THRESHOLD = 0.1

def run(seed, num_time_steps, n_directions, step_size, delta_std, rollout_length, policy_type):

    # logdir
    logdir = 'esdata/es_shell/{}/seed_{}nd_{}stepsize_{}deltastd_{}policytype_{}'.format(name, seed, n_directions, step_size, delta_std, policy_type)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # timelimit
    timelimit = 50

    # build env
    envs, stats = make_gurobi_env(LOGDIR, [0], timelimit=timelimit)

    # set seed
    np.random.seed(seed)
    try:
        for env in envs:
            env.seed(seed)
    except:
        pass

    # build policy
    policy_param = {'numvars':stats['numvars'],
                    'ob_filter':'NoFilter'}
    if policy_type == 'linear':
        policy_param['rowembed'] = 10
        policy = LinearRowFeaturePolicy(policy_param)
    elif policy_type == 'mlp':
        policy_param['hsize'] = 64
        policy_param['numlayers'] = 2
        policy = MLPRowFeaturePolicy(policy_param)
    elif policy_type == 'attention':
        policy_param['hsize'] = 64
        policy_param['numlayers'] = 2
        policy_param['embed'] = 10
        policy_param['rowembed'] = 10
        policy = MLPRowFeatureAttenttionPolicy(policy_param)
    elif policy_type == 'attentionembed':
        policy_param['hsize'] = 5
        policy_param['numlayers'] = 2
        policy_param['embed'] = 2
        policy_param['rowembed'] = 1
        policy = MLPRowFeatureAttenttionEmbeddingPolicy(policy_param)
    elif policy_type == 'lstmembed':
        policy_param['hsize'] = 64
        policy_param['numlayers'] = 2
        policy_param['embed'] = 10
        policy_param['rowembed'] = 10
        policy = MLPRowFeatureLSTMEmbeddingPolicy(policy_param)
        #MLPRowFeatureAttenttionPolicy(policy_param)
    else:
        raise NotImplementedError
    optimizer = Adam(policy.get_weights(), step_size)
    
    # initialize record table
    timestep_sofar = 0
    rewards_record = []
    times_record = []
    clocktime = []

    # determine if we want to load
    # load if there is a params
    if os.path.isfile(logdir + '/params.npy'):
        load = True
    else:
        load = False

    if load:
        # load data
        rewards_record = list(np.load(logdir + '/rewards.npy'))
        times_record = list(np.load(logdir + '/times.npy'))
        timestep_sofar = times_record[-1]
        params, mu, std = np.load(logdir + '/params.npy')
        policy.update_weights(params)
        print('setting filter')
        if hasattr(policy.observation_filter, 'mu'):
            policy.observation_filter.mu = mu
        if hasattr(policy.observation_filter, 'std'):
            policy.observation_filter.mu = std
        adam_v = np.load(logdir + '/adam_v.npy')
        adam_m = np.load(logdir + '/adam_m.npy')
        optimizer.v = adam_v
        optimizer.m = adam_m
        clocktime = list(np.load(logdir + '/clocktime.npy'))

    # training loop
    while timestep_sofar <= num_time_steps:

        time_start = time.time()

        original_weights = policy.get_weights().copy()

        # keep record of training rewards and epsilon
        epsilon_table = []
        train_rewards_table = []

        for i in range(n_directions):
            # sample n_directions noise
            epsilon = np.random.randn(*policy.get_weights().shape) * delta_std

            # update weights
            policy.update_weights(original_weights + epsilon)

            # rollout
            rewards, times = rollout_envs(envs=envs, policy=policy, num_rollouts=1, rollout_length=rollout_length, gamma=1.0)
            timestep_sofar += np.sum(times)

            # record rewards and epsilon
            epsilon_table.append(epsilon)
            train_rewards_table.append(np.mean(rewards))

        # acumulate gradients
        epsilon_table = np.array(epsilon_table)
        train_rewards_table = np.array(train_rewards_table)
        train_rewards_table = (train_rewards_table - np.mean(train_rewards_table)) / (np.std(train_rewards_table) + 1e-8)

        grad = np.mean(epsilon_table * train_rewards_table[:,np.newaxis], axis=0) / delta_std

        # assign back the original params
        policy.update_weights(original_weights)

        # update
        w = policy.get_weights() - optimizer._compute_step(grad.flatten()).reshape(policy.get_weights().shape)
        policy.update_weights(w)

        # eval rewards
        evaluated_rewards, _ = rollout_envs(envs=envs, policy=policy, num_rollouts=10, rollout_length=timelimit, gamma=1.0)
        #evaluated_rewards, _ = rollout_evaluate(envs=envs_eval, policy=policy, num_rollouts=10, rollout_length=timelimit, gamma=1.0)

        print(timestep_sofar)
        compute_stats(evaluated_rewards)

        # record results
        rewards_record.append(np.mean(evaluated_rewards))
        times_record.append(timestep_sofar)

        time_end = time.time()

        clocktime.append(time_end - time_start)

        # save everything
        np.save(logdir + '/rewards', rewards_record)
        np.save(logdir + '/times', times_record)
        np.save(logdir + '/params', policy.get_weights_plus_stats())
        np.save(logdir + '/adam_v', optimizer.v)
        np.save(logdir + '/adam_m', optimizer.m)
        np.save(logdir + '/clocktime', clocktime)
        #np.save(logdir + '/eval_rewards', evaluated_rewards)

        # terminate the program when the memory exceeds threshold
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss  # in bytes             
        available_mem = psutil.virtual_memory().available
        ratio = mem / (0.0 + available_mem)
        print('mem ratio', ratio)
        if ratio >= MEM_THRESHOLD:
            print('terminating due to memory threshold')
            break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_time_steps', '-n', type=int, default=int(10**9))
    parser.add_argument('--n_directions', '-nd', type=int, default=10)
    parser.add_argument('--step_size', '-s', type=float, default=0.01)
    parser.add_argument('--delta_std', '-std', type=float, default=0.02)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='attention')

    args = parser.parse_args()
    params = vars(args)

    run(seed=params['seed'],num_time_steps=params['num_time_steps'],
    	n_directions=params['n_directions'],step_size=params['step_size'],delta_std=params['delta_std'],
    	rollout_length=params['rollout_length'], policy_type=params['policy_type'])
