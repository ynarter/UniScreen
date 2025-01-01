import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt

###
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
    events_time = params['mean_event_times'] #Tn
    events_time = events_time + .1 * jax.random.normal(subkey, shape=(x.size,))
    events_time = np.where(events, events_time, np.inf)
    #print("Event times:", events_time)
    """ jax.debug.print("Event times: {}", events_time)
    jax.debug.print("Size (Event times): {}", events_time.shape)
    jax.debug.print("\n") """

    key, subkey = jax.random.split(key)
    death_time = params['mean_death_time']
    death_time = death_time + .1 * jax.random.normal(subkey)
    #print("Death times:", death_time)
    """ jax.debug.print("Death times: {}", death_time)
    jax.debug.print("Size (Death times): {}", death_time.shape)
    jax.debug.print("\n")
     """
    tau_k = jax.random.randint(key, shape=(2,), minval=1, maxval=6)
    """
    delta_k = np.zeros(params['sample_sizes'].shape)
    delta_k = delta_k.at[0, ::tau_k[0]].set(1)
    delta_k = delta_k.at[1, ::tau_k[1]].set(1)   
    """

    
    
    SAMPLE_STD = .5
    key, subkey = jax.random.split(key)
    samples = events[:,None] + SAMPLE_STD * jax.random.normal(subkey, shape=params['sample_sizes'].shape)
    #samples = samples * delta_k 
    #print("Samples (y):", samples)
    """ jax.debug.print("Samples (y): {}", samples)
    jax.debug.print("Size (Samples): {}", samples.shape)
    jax.debug.print("\n") """

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
    """ jax.debug.print("Posteriors: {}", posteriors)
    jax.debug.print("Size (Posteriors): {}", posteriors.shape)
    jax.debug.print("\n") """

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
    """ jax.debug.print("Diagnoses: {}", diagnoses)
    jax.debug.print("Size (Diagnoses): {}", diagnoses.shape)
    jax.debug.print("\n")
    jax.debug.print("Diagnoses time: {}", diagnoses_time)
    jax.debug.print("Size (Diagnoses time): {}", diagnoses_time.shape)
    jax.debug.print("\n") """

    # GAIN-COST:
    # assuming x.size == 2

    events_time_diag = np.where(diagnoses_time < events_time, np.inf, events_time) #T*
    #print("Events diagnoses time:", events_time_diag)
    """ jax.debug.print("Diagnoses time: {}", diagnoses_time)
    jax.debug.print("Size (Diagnoses time): {}", diagnoses_time.shape)
    jax.debug.print("\n") """
    
    survival_00 = np.minimum(death_time, np.minimum(events_time[0], events_time[1]))
    survival_10 = np.minimum(death_time, np.minimum(events_time_diag[0], events_time[1]))
    survival_01 = np.minimum(death_time, np.minimum(events_time[0], events_time_diag[1]))
    survival_11 = np.minimum(death_time, np.minimum(events_time_diag[0], events_time_diag[1]))
    
    gain_00 = survival_00 - survival_00  # = 0
    gain_10 = survival_10 - survival_00
    gain_01 = survival_01 - survival_00
    gain_11 = survival_11 - survival_00

    samples_time = np.minimum(diagnoses_time, params['sample_sizes'].sum(axis=-1))
    
    """ jax.debug.print("Survival time (00): {}", survival_00)
    jax.debug.print("Survival time (10): {}", survival_10)
    jax.debug.print("Survival time (01): {}", survival_01)
    jax.debug.print("Survival time (11): {}", survival_11)
    jax.debug.print("\n")
    
    jax.debug.print("Samples time: {}", samples_time)
    jax.debug.print("Size (Samples time): {}", samples_time.shape)
    jax.debug.print("\n") """

    #tau_k = params["sampling_freq"]
    #OR: 
     #for each patient, generate a random sampling freq for two diseases
    ck = params['screening_costs']
    
    cost_00 = 0
    cost_10 = ck[0] * np.floor(survival_10 / tau_k[0])
    cost_01 = ck[1] * np.floor(survival_01 / tau_k[1])
    cost_11 = ck[0] * np.floor(survival_11 / tau_k[0]) + ck[1] * np.floor(survival_11 / tau_k[1])
    

    """
    cost_00 = 0
    cost_10 = np.floor(np.minimum(survival_10, samples_time[0]))
    cost_01 = np.floor(np.minimum(survival_01, samples_time[1]))
    cost_11 = 0
    cost_11 = cost_11 + np.floor(np.minimum(survival_11, samples_time[0]))
    cost_11 = cost_11 + np.floor(np.minimum(survival_11, samples_time[1]))  
    """

    

    return (
        np.array([gain_00, gain_10, gain_01, gain_11]), 
        np.array([cost_00, cost_10, cost_01, cost_11]))

###

#N_INNER = 2_000
#N_OUTER = 10_000

N_INNER = 20
#N_OUTER = 100

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
params['mean_death_time'] = 70.
params['mean_event_times'] = np.array([60., 50.])
params['sample_sizes'] = np.ones((2, 10))
# params['sample_perrors'] = np.array([.2, .2])
params['diagnosis_threshold'] = .95
params['screening_costs'] = np.array([1., 1.]) #screening costs ck, will be modified accordingly.
#params['sampling_freq'] = delta_k, assume now as a sampling with fixed frequency. 

###

"""
key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key)
xs = jax.random.beta(subkey, 1, 1, shape=(N_OUTER, 2))
#np1.save("res/xs.npy", xs)
"""


#10x10 joint pmf for x1 and x2
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)

#joint pmf, ASSUME uniform distribution for now
joint_pmf = np.ones((100, 100)) / (100*100)

joint_pmf /= joint_pmf.sum()
#print("Joint pmf for x1 and x2:", joint_pmf)

def sample_from_pmf(pmf, key, N):
    pmf_flat = pmf.flatten()
    idx = jax.random.choice(key, np.arange(pmf.size), p=pmf_flat, shape=(N,)) #make a random choice, given the sample number N
    #print(idx)
    x1_idx = idx // pmf.shape[1] #determine the row/column
    x2_idx = idx % pmf.shape[1]
    x1_vals = x1[x1_idx]
    x2_vals = x2[x2_idx]
    xs = np.stack([x1_vals, x2_vals], axis=-1)
    return xs



# Updated part of the original simulation code
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

N = 3000
xs = sample_from_pmf(joint_pmf, subkey, N)
#print(xs)


key, subkey = jax.random.split(key)
gains, costs = simulate_batch(xs, subkey, params)

"""
#np1.save("res/gains.npy", gains)
#np1.save("res/costs.npy", costs)
"""


####################################################################################################################################

# Monte Carlo Process
def monte_carlo_simulation(num_iterations, xs, key, params, gamma_values, joint_pmf):
    """
    Perform Monte Carlo simulations by trying different values of gamma, x1, and x2.
    Returns the expected gains and costs for each patient.
    """
    # Initialize accumulators for gains and costs
    N = xs.shape[0]
    total_gains = np.zeros((N, 4))  # For policies 00, 10, 01, 11
    total_costs = np.zeros((N, 4))
    
    for _ in range(num_iterations):
        # Sample gamma, x1, and x2 for this iteration
        key, subkey = jax.random.split(key)
        gamma = jax.random.choice(subkey, gamma_values)
        #gamma = jax.random.choice(subkey, gamma_values, shape=(xs.shape[0],))  # random gamma value for each patient
        params['diagnosis_threshold'] =  gamma
        #print(params)
        
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        xs = sample_from_pmf(joint_pmf, subkey, N)
        
        # Run the simulation
        key, subkey = jax.random.split(key)
        gains, costs = simulate_batch(xs, subkey, params)
        
        # Accumulate the results
        total_gains += gains
        total_costs += costs

    # Compute expected values by averaging over the iterations
    expected_gains = total_gains / num_iterations
    expected_costs = total_costs / num_iterations
    
    print("Expected gains:", expected_gains)
    print("Expected costs:", expected_costs)

    return expected_gains, expected_costs


num_iterations = 300  # Number of Monte Carlo iterations
gamma_values = np.linspace(0.5, 1.0, 20)  # Range of gamma values to try


# Monte Carlo simulation to get expected gains and costs
key = jax.random.PRNGKey(0)  # Random key for sampling
expected_gains, expected_costs = monte_carlo_simulation(num_iterations, xs, key, params, gamma_values, joint_pmf)


def optimize_p1234(gain_to_cost, expected_gains, expected_costs):
    """
    Use convex optimization to determine the optimal p1, p2, p3, p4 for each patient,
    representing the probabilities for policies 00, 10, 01, 11.
    """
    num_patients = xs.shape[0]
    
    #optimization variables for p1, p2, p3, p4
    p1 = cp.Variable(num_patients)  # Probability for policy 00
    p2 = cp.Variable(num_patients)  # Probability for policy 10
    p3 = cp.Variable(num_patients)  # Probability for policy 01
    p4 = cp.Variable(num_patients)  # Probability for policy 11
    
    #ensure that the probabilities sum to 1 for each patient
    constraints = [
        p1 + p2 + p3 + p4 == 1,
        p1 >= 0, p2 >= 0, p3 >= 0, p4 >= 0  #ensure probabilities are non-negative
    ]
    
    #expected gains and costs
    #total_expected_gain = p1 * expected_gains[:, 0] + p2 * expected_gains[:, 1] + p3 * expected_gains[:, 2] + p4 * expected_gains[:, 3]
    #total_expected_cost = p1 * expected_costs[:, 0] + p2 * expected_costs[:, 1] + p3 * expected_costs[:, 2] + p4 * expected_costs[:, 3]
    
    total_expected_gain = cp.multiply(p1, expected_gains[:, 0]) + cp.multiply(p2, expected_gains[:, 1]) + cp.multiply(p3, expected_gains[:, 2]) + cp.multiply(p4, expected_gains[:, 3])
    total_expected_cost = cp.multiply(p1, expected_costs[:, 0]) + cp.multiply(p2, expected_costs[:, 1]) + cp.multiply(p3, expected_costs[:, 2]) + cp.multiply(p4, expected_costs[:, 3])
    
    
    #total_expected_gain = p1 * expected_gains[:, [0,1]] + p2 * expected_gains[:, [2,3]] + p3 * expected_gains[:, [0,2]] + p4 * expected_gains[:, [1,3]]
    #total_expected_cost = p1 * expected_costs[:, [0,1]] + p2 * expected_costs[:, [2,3]] + p3 * expected_costs[:, [0,2]] + p4 * expected_costs[:, [1,3]]
    #total_expected_gain = p1 * gains[:, 0] + p2 * gains[:, 1] + p3 * gains[:, 2] + p4 * gains[:, 3]
    #print(total_expected_cost)
    #total_cost = np1.sum(expected_costs)
    
    constraints.append( cp.sum(total_expected_cost)/num_patients <= 40 ) #given a B value for budget constraint
    
    # Define the objective to maximize the net gain-to-cost ratio
    #objective = cp.Maximize(cp.sum(expected_gains * gain_to_cost - expected_costs)) #MAY BE MODIFIED!!
    objective = cp.Maximize(cp.sum(total_expected_gain))
    
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the optimal values of p1, p2, p3, and p4
    return p1.value, p2.value, p3.value, p4.value

#optimize p1, p2, p3, p4 values
p1_optimal, p2_optimal, p3_optimal, p4_optimal = optimize_p1234(gain_to_cost=6.375, 
                                                                expected_gains=expected_gains, 
                                                                expected_costs=expected_costs)


def get_policy_from_p1234(p1_optimal, p2_optimal, p3_optimal, p4_optimal):
    """
    Determine the final policy for each patient using the optimal p1, p2, p3, and p4 values.
    """
    policy = []
    
    for i in range(xs.shape[0]):
        #construct PMF for the four possible policies using the optimal p1, p2, p3, and p4
        pmf = np.array([p1_optimal[i], p2_optimal[i], p3_optimal[i], p4_optimal[i]])
        
        #select the policy with the highest probability
        optimal_policy = np.argmax(pmf)
        policy.append(optimal_policy)

    return np.array(policy)

# Get the final policy based on the optimal p1, p2, p3, and p4
policy = get_policy_from_p1234(p1_optimal, p2_optimal, p3_optimal, p4_optimal)

# Plot the resulting policy assignments
def plot_policy(policy):
    plt.scatter(xs[policy == 0, 0], xs[policy == 0, 1], color='silver', s=50, label='00')
    plt.scatter(xs[policy == 1, 0], xs[policy == 1, 1], color='tab:pink', s=50, label='10')
    plt.scatter(xs[policy == 2, 0], xs[policy == 2, 1], color='tab:purple', s=50, label='01')
    plt.scatter(xs[policy == 3, 0], xs[policy == 3, 1], color='tab:red', s=50, label='11')

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot the optimized policy results
plot_policy(policy)