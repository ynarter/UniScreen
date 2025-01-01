import jax
import jax.numpy as np
import numpy as np1
import cvxpy as cp
from jax.scipy.special import logsumexp

def _simulate_singlekey(x, key, params):
    # Simulating environment events and agent behavior
    key, subkey = jax.random.split(key)
    events = jax.random.uniform(subkey, shape=(x.size,))
    events = events < x

    key, subkey = jax.random.split(key)
    events_time = params['mean_event_times']
    events_time = events_time + .1 * jax.random.normal(subkey, shape=(x.size,))
    events_time = np.where(events, events_time, np.inf)

    key, subkey = jax.random.split(key)
    death_time = params['mean_death_time']
    death_time = death_time + .1 * jax.random.normal(subkey)

    SAMPLE_STD = .5
    key, subkey = jax.random.split(key)
    samples = events[:, None] + SAMPLE_STD * jax.random.normal(subkey, shape=params['sample_sizes'].shape)

    # Posteriors
    _log_prior = np.stack((np.log(x), np.log(1-x)), axis=0)
    _log_likelihoods = np.stack((
        np.cumsum(-.5 * ((samples - 1) / SAMPLE_STD) ** 2, axis=-1),
        np.cumsum(-.5 * (samples / SAMPLE_STD) ** 2, axis=-1)), axis=0)
    _log_posteriors = np.concatenate((_log_prior[:, :, None], _log_prior[:, :, None] + _log_likelihoods), axis=-1)
    posteriors = np.exp(_log_posteriors[0] - logsumexp(_log_posteriors, axis=0))

    diagnoses = np.any(posteriors > params['diagnosis_threshold'], axis=-1)
    diagnoses_time = np.where(diagnoses, np.argmax(posteriors > params['diagnosis_threshold'], axis=-1), np.inf)

    return diagnoses_time

# New: Set up the convex optimization problem with CVXPY
def optimize_diagnosis_policy(xs, params):
    N, K = xs.shape  # N diseases, K screening targets
    T0 = params['max_survival_time']  # Maximum survival time
    B = params['budget']  # Total budget
    c = params['cost_per_sample']  # Cost per screening sample

    # Decision variables
    delta = cp.Variable((N, K), boolean=True)  # Screening policy
    theta_hat = cp.Variable(N, boolean=True)  # Diagnostic policy

    # Objective: Maximize survival time
    survival_time = cp.Variable()
    objective = cp.Maximize(survival_time)

    # Constraints
    constraints = []
    
    # Budget constraint
    constraints.append(cp.sum(cp.multiply(c, delta)) <= B)

    # Error rate constraint (false positives)
    error_rate = params['alpha']  # Maximum false positive rate
    for n in range(N):
        constraints.append(cp.sum(theta_hat[n] * (1 - xs[:, n])) <= error_rate[n])

    # Survival time constraints
    for n in range(N):
        constraints.append(survival_time <= cp.min(params['mean_event_times'][n] - theta_hat[n]))

    # Solve optimization problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return delta.value, theta_hat.value

# Parameters
params = {
    'mean_event_times': np.array([9., 8.]),
    'mean_death_time': 10.,
    'sample_sizes': np.ones((2, 10)),
    'diagnosis_threshold': .95,
    'max_survival_time': 15,
    'budget': 100,
    'cost_per_sample': np.array([1, 1]),
    'alpha': np.array([0.05, 0.05])  # False positive constraints
}

# Sample data
N_OUTER = 10
key = jax.random.PRNGKey(0)
xs = jax.random.beta(key, 1, 1, shape=(N_OUTER, 2))

# Run optimization
delta_opt, theta_hat_opt = optimize_diagnosis_policy(xs, params)

# Display optimized policy
print("Optimized screening policy:", delta_opt)
print("Optimized diagnostic policy:", theta_hat_opt)
