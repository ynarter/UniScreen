import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import beta
from collections import Counter
import os

##########INDEPENDENT MODEL:

def _simulate_singlekey_indep(x, key, params): 
    # ENVIRONMENT:

    key, subkey = jax.random.split(key)
    events_indep = jax.random.uniform(subkey, shape=(x.size,))
    events_indep = events_indep < x

    key, subkey = jax.random.split(key)
    events_time_indep = params['mean_event_times']  # Tn
    events_time_indep = events_time_indep + 3 * jax.random.normal(subkey, shape=(x.size,))
    #events_time_indep = jax.random.exponential(subkey, shape=(x.size,)) * events_time_indep
    events_time_indep = np.where(events_indep, events_time_indep, np.inf)

    key, subkey = jax.random.split(key)
    death_time_indep = params['mean_death_time']
    #death_time_indep = death_time_indep + .1 * jax.random.normal(subkey)

    tau_k_indep = params['sampling_freq']
    SAMPLE_STD_INDEP = .5
    key, subkey = jax.random.split(key)
    samples_indep = events_indep[:, None] + SAMPLE_STD_INDEP * jax.random.normal(subkey, shape=params['sample_sizes'].shape)

    # AGENT:
    _log_prior_indep = np.stack((np.log(x), np.log(1-x)), axis=0)
    _log_likelihoods_indep = np.stack((
        np.cumsum(-.5 * ((samples_indep - 1) / SAMPLE_STD_INDEP)**2, axis=-1),
        np.cumsum(-.5 * (samples_indep / SAMPLE_STD_INDEP)**2, axis=-1)), axis=0)
    _log_posteriors_indep = np.concatenate((_log_prior_indep[:, :, None], _log_prior_indep[:, :, None] + _log_likelihoods_indep), axis=-1)
    posteriors_indep = np.exp(_log_posteriors_indep[0] - logsumexp(_log_posteriors_indep, axis=0))

    gamma = params['diagnosis_threshold']
    
    diagnoses_indep = np.any(posteriors_indep > gamma, axis=-1)
    diagnoses_time_indep = np.where(diagnoses_indep, np.argmax(posteriors_indep > gamma, axis=-1), np.inf)

    # GAIN-COST:
    events_time_diag_indep = np.where(diagnoses_time_indep < events_time_indep, np.inf, events_time_indep)  # T*

    survival_0_1_indep = np.minimum(death_time_indep, events_time_indep[0])
    survival_1_1_indep = np.minimum(death_time_indep, events_time_diag_indep[0])
    survival_0_2_indep = np.minimum(death_time_indep, events_time_indep[1])
    survival_1_2_indep = np.minimum(death_time_indep, events_time_diag_indep[1])

    samples_time_indep = np.minimum(diagnoses_time_indep, params['sample_sizes'].sum(axis=-1))


    tau_k_indep = params["sampling_freq"]
    ck_indep = params['screening_costs']
    
    
    cost_0_1_indep = 0
    cost_1_1_indep = ck_indep[0] * np.floor(np.minimum(survival_1_1_indep, samples_time_indep[0])) #screening first disease
    cost_0_2_indep = 0
    cost_1_2_indep = ck_indep[1] * np.floor(np.minimum(survival_1_2_indep, samples_time_indep[1])) #screening second disease

    return (
        np.array([survival_0_1_indep, survival_1_1_indep, survival_0_2_indep, survival_1_2_indep]), 
        np.array([cost_0_1_indep, cost_1_1_indep, cost_0_2_indep, cost_1_2_indep])
    )

_simulate_multikey_indep = jax.vmap(_simulate_singlekey_indep, in_axes=(None, 0, None))
_simulate_multikey_indep = jax.jit(_simulate_multikey_indep)

@jax.jit
def simulate_indep(x, key, params):
    keys = jax.random.split(key, N_INNER)
    gains, costs = _simulate_multikey_indep(x, keys, params)
    mean_gain = gains.mean(axis=0)
    mean_cost = costs.mean(axis=0)
    return mean_gain, mean_cost

_simulate_batch_indep = jax.vmap(simulate_indep, in_axes=(0, 0, None))
_simulate_batch_indep = jax.jit(_simulate_batch_indep)

@jax.jit
def simulate_batch_indep(xs, key, params):
    keys = jax.random.split(key, xs.shape[0])
    return _simulate_batch_indep(xs, keys, params)

N_INNER = 200
N = 10000

params = dict()
params['mean_death_time'] = 40. #T0
params['mean_event_times'] = np.array([30, 30]) #Tn's
params['sample_sizes'] = np.ones((1, 40))
# params['sample_perrors'] = np.array([.2, .2])
params['diagnosis_threshold'] = .95
params['screening_costs'] = np.array([1, 1]) #screening costs ck, will be modified accordingly.
params['sampling_freq'] = np.array([5., 5.]) #assume now as a sampling with fixed frequency. 

###


def monte_carlo_simulation_indep(num_iterations, xs, key, params, save_dir="results_indep"):
    """
    Monte Carlo simulation for the independent model.
    Saves each iteration's gains and costs in separate .npy files.
    Computes and prints mean, variance, and std at the end.
    """
    os.makedirs(save_dir, exist_ok=True)

    N = xs.shape[0]
    all_gains_list = []
    all_costs_list = []

    for i in range(num_iterations):

        key, subkey = jax.random.split(key)
        gains, costs = simulate_batch_indep(xs, subkey, params)

        gains_np = np1.array(gains)
        costs_np = np1.array(costs)

        np1.save(f"{save_dir}/gains_indep_iter_{i}.npy", gains_np)
        np1.save(f"{save_dir}/costs_indep_iter_{i}.npy", costs_np)

        all_gains_list.append(gains_np)
        all_costs_list.append(costs_np)

    all_gains = np1.stack(all_gains_list)
    all_costs = np1.stack(all_costs_list)

    mean_gains = np1.mean(all_gains, axis=0)
    var_gains = np1.var(all_gains, axis=0)
    std_gains = np1.std(all_gains, axis=0)

    mean_costs = np1.mean(all_costs, axis=0)
    var_costs = np1.var(all_costs, axis=0)
    std_costs = np1.std(all_costs, axis=0)

    """ print("Expected gains (mean):", mean_gains)
    print("Variance of gains:", var_gains)
    print("Standard deviation of gains:", std_gains)

    print("Expected costs (mean):", mean_costs)
    print("Variance of costs:", var_costs)
    print("Standard deviation of costs:", std_costs) """

    return mean_gains, mean_costs

num_iterations = 100  # Number of Monte Carlo iterations

# Parameters for the Beta distributions
a, b = 3, 5  # Parameters

x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
xs = np.stack([X1.ravel(), X2.ravel()], axis=-1)  # Create grid combinations
dx = x1[1] - x1[0]  # Interval width

pdf_x1 = beta.pdf(x1, a, b)
pdf_x2 = beta.pdf(x2, a, b)
joint_pmf = np.outer(pdf_x1, pdf_x2) * dx * dx

joint_pmf /= np.sum(joint_pmf)
pmf_x = joint_pmf.flatten()

#run Monte Carlo
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
expected_gains_indep, expected_costs_indep = monte_carlo_simulation_indep(num_iterations, xs, key, params)

#for the first disease
survival_disease_1 = np1.array([expected_gains_indep[:, 0], expected_gains_indep[:, 1]])  # survival_0_1 and survival_1_1
cost_disease_1 = np1.array([expected_costs_indep[:,0], expected_costs_indep[:,1]])  # cost_0_1 and cost_1_1

#for the second disease
survival_disease_2 = np1.array([expected_gains_indep[:,2], expected_gains_indep[:,3]])  # survival_0_2 and survival_1_2
cost_disease_2 = np1.array([expected_costs_indep[:,2], expected_costs_indep[:,3]]) # cost_0_2 and cost_1_2

#Save the data:
folder_path = "comparison/independent_data"
os.makedirs(folder_path, exist_ok=True)

np.save(os.path.join(folder_path, "survival_disease_1.npy"), survival_disease_1)
np.save(os.path.join(folder_path, "cost_disease_1.npy"), cost_disease_1)
np.save(os.path.join(folder_path, "survival_disease_2.npy"), survival_disease_2)
np.save(os.path.join(folder_path, "cost_disease_2.npy"), cost_disease_2)

print(f"Data saved successfully in '{folder_path}'!")

