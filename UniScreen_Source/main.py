import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt

###
def _simulate_singlekey(x, key, params):

    # ENVIRONMENT:

    key, subkey = jax.random.split(key)
    events = jax.random.uniform(subkey, shape=(x.size,))
    events = events < x
    #print("Events:", events)


    key, subkey = jax.random.split(key)
    events_time = params['mean_event_times']
    events_time = events_time + .1 * jax.random.normal(subkey, shape=(x.size,))
    events_time = np.where(events, events_time, np.inf)
    #print("Event times:", events_time)


    key, subkey = jax.random.split(key)
    death_time = params['mean_death_time']
    death_time = death_time + .1 * jax.random.normal(subkey)
    #print("Death times:", death_time)

    
    
    SAMPLE_STD = .5
    key, subkey = jax.random.split(key)
    samples = events[:,None] + SAMPLE_STD * jax.random.normal(subkey, shape=params['sample_sizes'].shape)
    #print("Samples (y):", samples)


    # key, subkey = jax.random.split(key)
    # samples = jax.random.uniform(subkey, shape=params['sample_sizes'].shape)
    # samples = samples < np.where(events, 1 - params['sample_perrors'], params['sample_perrors'])[:,None]

    # AGENT:

    _log_prior = np.stack((np.log(x), np.log(1-x)), axis=0)
    _log_likelihoods = np.stack((
        np.cumsum(-.5 * ((samples - 1) / SAMPLE_STD)**2, axis=-1),
        np.cumsum(-.5 * (samples / SAMPLE_STD)**2, axis=-1)), axis=0)
    _log_posteriors = np.concatenate((_log_prior[:,:,None], _log_prior[:,:,None] + _log_likelihoods), axis=-1)
    posteriors = np.exp(_log_posteriors[0] - logsumexp(_log_posteriors, axis=0))
    #print("Posteriors:", posteriors)


    # _log_prior = np.stack((np.log(x), np.log(1-x)), axis=0)
    # _log_error = np.log(params['sample_perrors'])[:,None]
    # _log_noerror = np.log(1 - params['sample_perrors'])[:,None]
    # _log_likelihoods = np.stack((
    #     np.cumsum(np.where(samples, _log_noerror, _log_error), axis=-1),
    #     np.cumsum(np.where(samples, _log_error, _log_noerror), axis=-1)), axis=0)
    # _log_posteriors = np.concatenate((_log_prior[:,:,None], _log_prior[:,:,None] + _log_likelihoods), axis=-1)
    # posteriors = np.exp(_log_posteriors[0] - logsumexp(_log_posteriors, axis=0))

    diagnoses = np.any(posteriors > params['diagnosis_threshold'], axis=-1)
    diagnoses_time = np.where(diagnoses, np.argmax(posteriors > params['diagnosis_threshold'], axis=-1), np.inf)
    #print("Diagnoses:", diagnoses)
    #print("Diagnoses time:", diagnoses_time)

    # GAIN-COST:
    # assuming x.size == 2

    events_time_diag = np.where(diagnoses_time < events_time, np.inf, events_time)
    #print("Events diagnoses time:", events_time_diag)
    
    survival_00 = np.minimum(death_time, np.minimum(events_time[0], events_time[1]))
    survival_10 = np.minimum(death_time, np.minimum(events_time_diag[0], events_time[1]))
    survival_01 = np.minimum(death_time, np.minimum(events_time[0], events_time_diag[1]))
    survival_11 = np.minimum(death_time, np.minimum(events_time_diag[0], events_time_diag[1]))
    
    gain_00 = survival_00 - survival_00  # = 0
    gain_10 = survival_10 - survival_00
    gain_01 = survival_01 - survival_00
    gain_11 = survival_11 - survival_00

    samples_time = np.minimum(diagnoses_time, params['sample_sizes'].sum(axis=-1))
    

    cost_00 = 0
    cost_10 = np.floor(np.minimum(survival_10, samples_time[0]))
    cost_01 = np.floor(np.minimum(survival_01, samples_time[1]))
    cost_11 = 0
    cost_11 = cost_11 + np.floor(np.minimum(survival_11, samples_time[0]))
    cost_11 = cost_11 + np.floor(np.minimum(survival_11, samples_time[1]))

    return (
        np.array([gain_00, gain_10, gain_01, gain_11]), 
        np.array([cost_00, cost_10, cost_01, cost_11]))

###

#N_INNER = 2_000
#N_OUTER = 10_000

N_INNER = 2000
N_OUTER = 10000

_simulate_multikey = jax.vmap(_simulate_singlekey, in_axes=(None,0,None))
_simulate_multikey = jax.jit(_simulate_multikey)

@jax.jit
def simulate(x, key, params):
    keys = jax.random.split(key, N_INNER)
    gains, costs = _simulate_multikey(x, keys, params)
    mean_gain = gains.mean(axis=0)
    mean_cost = costs.mean(axis=0)
    return mean_gain, mean_cost

_simulate_batch = jax.vmap(simulate, in_axes=(0,0,None))
_simulate_batch = jax.jit(_simulate_batch)

@jax.jit
def simulate_batch(xs, key, params):
    keys = jax.random.split(key, xs.shape[0])
    return _simulate_batch(xs, keys, params)

###

params = dict()
params['mean_death_time'] = 10.
params['mean_event_times'] = np.array([9., 8.])
params['sample_sizes'] = np.ones((2, 10))
# params['sample_perrors'] = np.array([.2, .2])
params['diagnosis_threshold'] = .95

###

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
xs = jax.random.beta(subkey, 1, 1, shape=(N_OUTER, 2))


key, subkey = jax.random.split(key)
gains, costs = simulate_batch(xs, subkey, params)


print(gains.shape)
print(costs.shape)


def get_policy(gain_to_cost):
    policy = np.argmax(gains * gain_to_cost - costs, axis=-1)
    budget = np.array([costs[i, policy[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
    return policy, budget

policy, budget = get_policy(gain_to_cost=10)
print(budget)

# 00 10 01 11
def get_policy_conditionalx0(gain_to_cost):
    _gains = gains[:,[0,1]]
    _costs = costs[:,[0,1]]
    policy = np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    return policy

def get_policy_conditionalx1(gain_to_cost):
    _gains = gains[:,[2,3]]
    _costs = costs[:,[2,3]]
    policy = 2 + np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    return policy

def get_policy_conditional0x(gain_to_cost):
    _gains = gains[:,[0,2]]
    _costs = costs[:,[0,2]]
    policy = 2 * np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    return policy

def get_policy_conditional1x(gain_to_cost):
    _gains = gains[:,[1,3]]
    _costs = costs[:,[1,3]]
    policy = 1 + 2 * np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    return policy

policy_conditionalx0 = get_policy_conditionalx0(gain_to_cost=10)
policy_conditionalx1 = get_policy_conditionalx1(gain_to_cost=10)
policy_conditional0x = get_policy_conditional0x(gain_to_cost=10)
policy_conditional1x = get_policy_conditional1x(gain_to_cost=10)

policy_conditionalx0 = (policy_conditionalx0 == 1)
policy_conditionalx1 = (policy_conditionalx1 == 3)
policy_conditional0x = (policy_conditional0x == 2)
policy_conditional1x = (policy_conditional1x == 3)

yea_0 = policy_conditionalx0 & policy_conditionalx1
nay_0 = ~policy_conditionalx0 & ~policy_conditionalx1
yea_1 = policy_conditional0x & policy_conditional1x
nay_1 = ~policy_conditional0x & ~policy_conditional1x

_00 = nay_0 & nay_1
_10 = yea_0 & nay_1
_01 = nay_0 & yea_1
_11 = yea_0 & yea_1

_11 = _11 | (yea_0 & policy_conditional1x)
_11 = _11 | (yea_1 & policy_conditionalx1)

_00 = _00 | (nay_0 & ~policy_conditional0x)
_00 = _00 | (nay_1 & ~policy_conditionalx0)

_10 = _10 | (yea_0 & ~policy_conditional1x)
_10 = _10 | (nay_1 & policy_conditionalx0)

_01 = _01 | (nay_0 & policy_conditional0x)
_01 = _01 | (yea_1 & ~policy_conditionalx0)

plt.scatter(xs[_00, 0], xs[_00, 1], color='silver', s=50)
plt.scatter(xs[_10, 0], xs[_10, 1], color='tab:pink', s=50)
plt.scatter(xs[_01, 0], xs[_01, 1], color='tab:purple', s=50)
plt.scatter(xs[_11, 0], xs[_11, 1], color='tab:red', s=50)

plt.axis('scaled')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.show()

plt.tight_layout()
plt.savefig('figs/points.pdf')

def get_policy_conditionalxx(gain_to_cost):
    _gains = gains[:,[0,3]]
    _costs = costs[:,[0,3]]
    policy = 3 * np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    return policy

#policy_conditionalxx = get_policy_conditionalxx(gain_to_cost=10)

def get_policy_independent(budget):

    for threshold0 in np.linspace(0, 1, 1000):
        _budget = costs[xs[:,0] > threshold0, 1].sum() / xs.shape[0]
        if _budget < budget / 2:
            break

    for threshold1 in np.linspace(0, 1, 1000):
        _budget = costs[xs[:,1] > threshold1, 2].sum() / xs.shape[0]
        if _budget < budget / 2:
            break

    policy = np.zeros(xs.shape[0])
    policy += np.where(xs[:,0] > threshold0, 1, 0)
    policy += np.where(xs[:,1] > threshold1, 2, 0)
    
    return policy, np.array([threshold0, threshold1])

#policy_independent, thresholds = get_policy_independent(budget)

def plot(policy):

    plt.scatter(xs[policy == 0, 0], xs[policy == 0, 1], color='silver', s=50)
    plt.scatter(xs[policy == 1, 0], xs[policy == 1, 1], color='tab:pink', s=50)
    plt.scatter(xs[policy == 2, 0], xs[policy == 2, 1], color='tab:purple', s=50)
    plt.scatter(xs[policy == 3, 0], xs[policy == 3, 1], color='tab:red', s=50)

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()

    plt.tight_layout()
    plt.show()
    plt.savefig('figs/points.pdf')

def plot_diff(policy, _policy):

    plt.scatter(xs[(policy == 0) & (_policy != 0), 0], xs[(policy == 0) & (_policy != 0), 1], color='silver', s=50)
    plt.scatter(xs[(policy == 1) & (_policy != 1), 0], xs[(policy == 1) & (_policy != 1), 1], color='tab:pink', s=50)
    plt.scatter(xs[(policy == 2) & (_policy != 2), 0], xs[(policy == 2) & (_policy != 2), 1], color='tab:purple', s=50)
    plt.scatter(xs[(policy == 3) & (_policy != 3), 0], xs[(policy == 3) & (_policy != 3), 1], color='tab:red', s=50)

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()

    plt.tight_layout()
    plt.savefig('figs/points.pdf')

# plot_diff(policy_independent, policy)

plot(policy)

# benefit = np.array([gains[i, policy[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
# spending = np.array([costs[i, policy[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
# print(benefit / spending)

# policy_independent = policy_independent.astype(int)

# benefit = np.array([gains[i, policy_independent[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
# spending = np.array([costs[i, policy_independent[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
# print(benefit / spending)