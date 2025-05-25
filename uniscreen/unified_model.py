import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import beta, norm
from collections import Counter
import os

############OUR MODEL:

def _simulate_singlekey(x, key, params):

    # ENVIRONMENT:

    key, subkey = jax.random.split(key)
    events = jax.random.uniform(subkey, shape=(x.size,))
    events = events < x
    #print("Events:", events)
    """     
    jax.debug.print("Events: {}", events)
    jax.debug.print("Size (Events): {}", events.shape)
    jax.debug.print("\n") 
    
    """

    key, subkey = jax.random.split(key)
    events_time = params['mean_event_times']  # Tn
    #events_time = jax.random.exponential(subkey, shape=(x.size,)) * events_time
    
    #mean_event_times = params['mean_event_times']  # Shape (1, 2), e.g., [mean_disease1, mean_disease2]

    # Compute the scale as 1 / mean_event_times
    #scale = 1 / mean_event_times

    events_time = events_time + 3 * jax.random.normal(subkey, shape=(x.size,))
    events_time = np.where(events, events_time, np.inf)
    #print("Event times:", events_time)
    """ jax.debug.print("Event times: {}", events_time)
    jax.debug.print("Size (Event times): {}", events_time.shape)
    jax.debug.print("\n") """

    key, subkey = jax.random.split(key)
    death_time = params['mean_death_time']
    #death_time = death_time + .1 * jax.random.normal(subkey)
    #print("Death times:", death_time)
    """ jax.debug.print("Death times: {}", death_time)
    jax.debug.print("Size (Death times): {}", death_time.shape)
    jax.debug.print("\n")
     """
    #tau_k = jax.random.randint(key, shape=(2,), minval=1, maxval=6)
    tau_k = params['sampling_freq']
    SAMPLE_STD = .5
    key, subkey = jax.random.split(key)
    samples = events[:,None] + SAMPLE_STD * jax.random.normal(subkey, shape=params['sample_sizes'].shape)
    #samples = samples * delta_k 
    #print("Samples (y):", samples)
    """ jax.debug.print("Samples (y): {}", samples)
    jax.debug.print("Size (Samples): {}", samples.shape)
    jax.debug.print("\n") """


    # AGENT:

    _log_prior = np.stack((np.log(x), np.log(1-x)), axis=0)
    _log_likelihoods = np.stack((
        np.cumsum(-.5 * ((samples - 1) / SAMPLE_STD)**2, axis=-1),
        np.cumsum(-.5 * (samples / SAMPLE_STD)**2, axis=-1)), axis=0)
    _log_posteriors = np.concatenate((_log_prior[:,:,None], _log_prior[:,:,None] + _log_likelihoods), axis=-1)
    posteriors = np.exp(_log_posteriors[0] - logsumexp(_log_posteriors, axis=0))

    gamma = params['diagnosis_threshold']
    
    diagnoses = np.any(posteriors > gamma, axis=-1)
    diagnoses_time = np.where(diagnoses, np.argmax(posteriors > gamma, axis=-1), np.inf)

    # GAIN-COST:
    # assuming x.size == 2

    events_time_diag = np.where(diagnoses_time < events_time, np.inf, events_time) #T*
    
    survival_00 = np.minimum(death_time, np.minimum(events_time[0], events_time[1]))
    survival_10 = np.minimum(death_time, np.minimum(events_time_diag[0], events_time[1]))
    survival_01 = np.minimum(death_time, np.minimum(events_time[0], events_time_diag[1]))
    survival_11 = np.minimum(death_time, np.minimum(events_time_diag[0], events_time_diag[1]))
    
    gain_00 = survival_00 - survival_00  # = 0
    gain_10 = survival_10 - survival_00
    gain_01 = survival_01 - survival_00
    gain_11 = survival_11 - survival_00

    samples_time = np.minimum(diagnoses_time, params['sample_sizes'].sum(axis=-1))

    tau_k = params["sampling_freq"]
    ck = params['screening_costs']
     
    
    cost_00 = 0
    cost_10 = ck[0] * np.floor(np.minimum(survival_10, samples_time[0]))
    cost_01 = ck[1] * np.floor(np.minimum(survival_01, samples_time[1]))
    cost_11 = 0
    cost_11 = cost_11 + ck[0] * np.floor(np.minimum(survival_11, samples_time[0]))
    cost_11 = cost_11 + ck[1] * np.floor(np.minimum(survival_11, samples_time[1]))
   
    
    """ return (
        np.array([survival_00, survival_10, survival_01, survival_11]), 
        np.array([cost_00, cost_10, cost_01, cost_11])) """
    
    return (
        np.array([survival_00, survival_10, survival_01, survival_11]), 
        np.array([cost_00, cost_10, cost_01, cost_11]),
        events_time[0],
        diagnoses_time[0],
        events[0])

###

N_INNER = 200
N = 10000

_simulate_multikey = jax.vmap(_simulate_singlekey, in_axes=(None,0,None))
_simulate_multikey = jax.jit(_simulate_multikey)

@jax.jit
def simulate(x, key, params):
    keys = jax.random.split(key, N_INNER)
    gains, costs, tn, t_star, thetas = _simulate_multikey(x, keys, params)
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
params['mean_death_time'] = 40 #T0
params['mean_event_times'] = np.array([30.1, 30.1]) #Tn means
params['sample_sizes'] = np.ones((1, 40)) #1,40, assuming annual screening
# params['sample_perrors'] = np.array([.2, .2])
params['diagnosis_threshold'] = .95
params['screening_costs'] = np.array([1, 1]) #screening costs cn, will be modified accordingly. 1-1
params['sampling_freq'] = np.array([5., 5.]) #assume a sampling with fixed frequency (for old implementation) 

###



def monte_carlo_simulation(num_iterations, xs, key, params, save_dir="results"):
    """
    Monte Carlo simulation saving each iteration's gains and costs as separate .npy files.
    Computes and prints mean, variance, and std at the end.
    """
    os.makedirs(save_dir, exist_ok=True)

    N = xs.shape[0]
    all_gains_list = []
    all_costs_list = []

    for i in range(num_iterations):
        key, subkey = jax.random.split(key)

        key, subkey = jax.random.split(key)
        gains, costs = simulate_batch(xs, subkey, params)

        # Convert to NumPy and save separately
        gains_np = np1.array(gains)
        costs_np = np1.array(costs)

        np1.save(f"{save_dir}/gains_iter_{i}.npy", gains_np)
        np1.save(f"{save_dir}/costs_iter_{i}.npy", costs_np)

        all_gains_list.append(gains_np)
        all_costs_list.append(costs_np)

    # Stack results for statistics
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


num_iterations = 100  #number of Monte Carlo iterations

# Parameters for the Beta distributions
a, b = 5, 5  # Parameters

x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
xs = np.stack([X1.ravel(), X2.ravel()], axis=-1)
dx = x1[1] - x1[0]

#compute the PDF values
pdf_x1 = beta.pdf(x1, a, b)
pdf_x2 = beta.pdf(x2, a, b)

joint_pmf = np.outer(pdf_x1, pdf_x2) * dx * dx
joint_pmf /= np.sum(joint_pmf)
pmf_x = joint_pmf.flatten()


key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

expected_gains, expected_costs = monte_carlo_simulation(num_iterations, xs, key, params)

folder_path = "comparison/unified_data"
os.makedirs(folder_path, exist_ok=True)

np.save(os.path.join(folder_path, "expected_gains.npy"), expected_gains)
np.save(os.path.join(folder_path, "expected_costs.npy"), expected_costs)

print(f"Data saved successfully in '{folder_path}'!")

folder_path_2 = "comparison/xs"
os.makedirs(folder_path_2, exist_ok=True)

np.save(os.path.join(folder_path_2, "xs.npy"), xs)
np.save(os.path.join(folder_path_2, "pmf_x.npy"), pmf_x)

print(f"Data saved successfully in '{folder_path_2}'!")
