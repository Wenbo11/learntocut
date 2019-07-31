import numpy as np 
from es.policies import LinearRowFeaturePolicy, MLPRowFeaturePolicy, MLPRowFeatureAttenttionPolicy, MLPRowFeatureAttenttionEmbeddingPolicy, MLPRowFeatureLSTMEmbeddingPolicy

def load_policy(seed, n_directions, step_size, delta_std, policy_type, numvars, logdir):
    # logdir
    #logdir = '../esdata/es_mem/randomip_n30m15_onlyobjreward/seed_{}nd_{}stepsize_{}deltastd_{}policytype_{}gamma_1.0'.format(seed, n_directions, step_size, delta_std, policy_type)
    #logdir = '../esdata/es_shell/randomip_n30m15/seed_{}nd_{}stepsize_{}deltastd_{}policytype_{}'.format(seed, n_directions, step_size, delta_std, policy_type)
    logdir = '{}/seed_{}nd_{}stepsize_{}deltastd_{}policytype_{}'.format(logdir, seed, n_directions, step_size, delta_std, policy_type)

    # build policy
    policy_param = {'numvars':numvars,
                    'ob_filter':'MeanStdFilter'}

    if policy_type == 'linear':
        policy = LinearRowFeaturePolicy(policy_param)
    elif policy_type == 'mlp':
        policy_param['hsize'] = 64
        policy_param['numlayers'] = 2
        policy = MLPRowFeaturePolicy(policy_param)
    elif policy_type == 'attention':
        policy_param['hsize'] = 64
        policy_param['numlayers'] = 2
        policy_param['embed'] = 10
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
    else:
        raise NotImplementedError

    if True:
        print('loading policy')
        params, mu, std = np.load(logdir + '/params.npy')
        policy.update_weights(params)
        print('setting filter')
        if hasattr(policy.observation_filter, 'mu'):
            policy.observation_filter.mu = mu
        if hasattr(policy.observation_filter, 'std'):
            policy.observation_filter.mu = std

    return policy