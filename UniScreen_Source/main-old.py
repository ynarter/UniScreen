import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt

###
def _simulate_singlekey(p, key, params):
    params_horizon = 10

    # ENVIRONMENT:

    key, subkey = jax.random.split(key)
    events = jax.random.uniform(subkey, shape=(p.size,))
    events = events < p

    key, subkey = jax.random.split(key)
    events_time = jax.random.exponential(subkey, shape=(p.size,))
    events_time = np.where(events, events_time / params['events_rate'], np.inf)

    key, subkey = jax.random.split(key)
    samples = jax.random.uniform(subkey, shape=(p.size,params_horizon))
    samples = samples < np.where(events, 1 - params['samples_perror'], params['samples_perror'])[:,None]

    # AGENT:

    _logits_err = np.log(params['samples_perror'])[:,None]
    _logits_noerr = np.log(1 - params['samples_perror'])[:,None]
    _logits_event = np.where(samples, _logits_noerr, _logits_err).cumsum(axis=-1)
    _logits_noevent = np.where(samples, _logits_err, _logits_noerr).cumsum(axis=-1)

    _logits = np.stack((np.log(p), np.log(1-p)), axis=0)
    _logits = _logits[:,:,None] + np.stack((_logits_event, _logits_noevent), axis=0)
    _p = np.exp(_logits - logsumexp(_logits, axis=0))
    # _p = np.concatenate((np.stack((p, 1-p), axis=0)[:,:,None], _p), axis=-1)

    diagnoses = np.any(_p > params['diagnosis_threshold'], axis=-1)
    diagnoses_time = np.where(diagnoses, 1 + np.argmax(_p > params['diagnosis_threshold'], axis=-1), np.inf)

    # GAIN:
    # assumes (p.size == 2)

    posdiag_time = diagnoses_time[0,:]
    posdiag_isontime = events & diagnoses[0,:] & (events_time > diagnoses_time[0,:])
    sample_count = np.minimum(params_horizon, diagnoses_time.min(axis=0))

    ###

    gain_10 = posdiag_isontime[0]
    gain_01 = posdiag_isontime[1]
    gain_11 = posdiag_isontime.sum()

    ###

    # _posdiag_isontime_10 = posdiag_isontime[0] & (~events[1] | (events_time[1] > posdiag_time[0]))
    # _posdiag_isontime_01 = posdiag_isontime[1] & (~events[0] | (events_time[0] > posdiag_time[1]))
    # _posdiag_isontime_11a = posdiag_isontime[0] & (~events[1] | (events_time[1] > posdiag_time[0]) | posdiag_isontime[1])
    # _posdiag_isontime_11b = posdiag_isontime[1] & (~events[0] | (events_time[0] > posdiag_time[1]) | posdiag_isontime[0])

    # gain_10 = _posdiag_isontime_10
    # gain_01 = _posdiag_isontime_01
    # gain_11 = (_posdiag_isontime_11a + _posdiag_isontime_11b)

    ###

    cost_10 = sample_count[0]
    cost_01 = sample_count[1]
    cost_11 = sample_count.sum()

    # gain_10 = 10 * gain_10 - cost_10
    # gain_01 = 10 * gain_01 - cost_01
    # gain_11 = 10 * gain_11 - cost_11

    return np.array([0, gain_10, gain_01, gain_11]), np.array([0, cost_10, cost_01, cost_11])

_simulate_multikey = jax.vmap(_simulate_singlekey, in_axes=(None,0,None))
_simulate_multikey = jax.jit(_simulate_multikey)

@jax.jit
def simulate(p, key, params):
    keys = jax.random.split(key, 1000)
    gains, costs = _simulate_multikey(p, keys, params)
    mean_gain = gains.mean(axis=0)
    mean_cost = costs.mean(axis=0)
    return mean_gain, mean_cost

_simulate_batch = jax.vmap(simulate, in_axes=(0,0,None))
_simulate_batch = jax.jit(_simulate_batch)

@jax.jit
def simulate_batch(ps, key, params):
    keys = jax.random.split(key, ps.shape[0])
    return _simulate_batch(ps, keys, params)

###

params = dict()

params['events_rate'] = np.array([.1, .1])
params['samples_perror'] = np.array([.2, .2])
params['diagnosis_threshold'] = .95

###

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
ps = jax.random.beta(subkey, 1, 1, shape=(10000,2))
np1.save('res/main-ps.npy', ps)

key, subkey = jax.random.split(key)
gains, costs = simulate_batch(ps, subkey, params)
np1.save('res/main-gains.npy', gains)
np1.save('res/main-costs.npy', costs)

###

plt.scatter(ps[:,0], ps[:,1], color='k')

plt.axis('scaled')
plt.xlim(0,1)
plt.ylim(0,1)
plt.grid()

plt.tight_layout()
plt.savefig('figs/points.pdf')
