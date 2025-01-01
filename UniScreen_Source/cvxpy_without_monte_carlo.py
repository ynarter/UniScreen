import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt

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
    #tau_k = jax.random.randint(key, shape=(2,), minval=1, maxval=6)
    tau_k = np.array([5., 5.])
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

N_INNER = 2000
#N_OUTER = 100
N = 10000

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


key = jax.random.PRNGKey(0)
#key, subkey = jax.random.split(key)
#xs = sample_from_pmf(joint_pmf, subkey, N)

key, subkey = jax.random.split(key)
xs = jax.random.beta(subkey, 1, 1, shape=(N, 2))

key, subkey = jax.random.split(key)
gains, costs = simulate_batch(xs, subkey, params)





"""
def optimize_p1234(gain_to_cost):
    #Use convex optimization to determine the optimal p1, p2, p3, p4 for each patient,
    #representing the probabilities for policies 00, 10, 01, 11.
    num_patients = xs.shape[0]
    
    p1 = cp.Variable(num_patients)  # Probability for policy 00
    p2 = cp.Variable(num_patients)  # Probability for policy 10
    p3 = cp.Variable(num_patients)  # Probability for policy 01
    p4 = cp.Variable(num_patients)  # Probability for policy 11
    
    #optimization variables for p1, p2, p3, p4
    p1 = cp.Variable()  # Probability for policy 00
    p2 = cp.Variable()  # Probability for policy 10
    p3 = cp.Variable()  # Probability for policy 01
    p4 = cp.Variable()  # Probability for policy 11
    
    #ensure that the probabilities sum to 1 for each patient
    constraints = [
        p1 + p2 + p3 + p4 == 1,
        p1 >= 0, p2 >= 0, p3 >= 0, p4 >= 0  #ensure probabilities are non-negative
    ]
    
    #expected gains and costs
    #expected_gains = p1 * gains[:, 0] + p2 * gains[:, 1] + p3 * gains[:, 2] + p4 * gains[:, 3]
    #expected_costs = p1 * costs[:, 0] + p2 * costs[:, 1] + p3 * costs[:, 2] + p4 * costs[:, 3]
    
    expected_gains = cp.multiply(p1, gains[:, 0]) + cp.multiply(p2, gains[:, 1]) + cp.multiply(p3, gains[:, 2]) + cp.multiply(p4, gains[:, 3])
    expected_costs = cp.multiply(p1, costs[:, 0]) + cp.multiply(p2, costs[:, 1]) + cp.multiply(p3, costs[:, 2]) + cp.multiply(p4, costs[:, 3])
    #total_cost = np.sum(expected_costs)
    
    constraints.append( cp.sum(expected_costs) / num_patients <= 40 ) #given a B value for budget constraint
    
    # Define the objective to maximize the net gain-to-cost ratio
    #objective = cp.Maximize(cp.sum(expected_gains * gain_to_cost - expected_costs)) #MAY BE MODIFIED!!
    objective = cp.Maximize(cp.sum(expected_gains))
    
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the optimal values of p1, p2, p3, and p4
    return p1.value, p2.value, p3.value, p4.value
"""


"""
def optimize_p1234(gain_to_cost, gains, costs):
    #Use convex optimization to determine the optimal p1, p2, p3, p4 for each patient,
    #representing the probabilities for policies 00, 10, 01, 11.
    #Solve the problem for each patient individually. 
    num_patients = gains.shape[0]  # Assuming 'gains' is a matrix with shape (num_patients, 4)
    
    # Store results for all patients
    p1_values = np1.zeros(num_patients)
    p2_values = np1.zeros(num_patients)
    p3_values = np1.zeros(num_patients)
    p4_values = np1.zeros(num_patients)

    # Loop through each patient (each row of the matrix)
    for i in range(num_patients):
        # Optimization variables for p1, p2, p3, p4 for the current patient
        p1 = cp.Variable()  # Probability for policy 00 (binary: 0 or 1)
        p2 = cp.Variable()  # Probability for policy 10 (binary: 0 or 1)
        p3 = cp.Variable()  # Probability for policy 01 (binary: 0 or 1)
        p4 = cp.Variable()  # Probability for policy 11 (binary: 0 or 1)
        
        # Constraints ensuring probabilities sum to 1 and are non-negative
        constraints = [
            p1 + p2 + p3 + p4 == 1,
            p1 >= 0, p2 >= 0, p3 >= 0, p4 >= 0
        ]
        
        # Expected gains and costs for the current patient
        expected_gains = cp.multiply(p1, gains[i, 0]) + cp.multiply(p2, gains[i, 1]) + cp.multiply(p3, gains[i, 2]) + cp.multiply(p4, gains[i, 3])
        expected_costs = cp.multiply(p1, costs[i, 0]) + cp.multiply(p2, costs[i, 1]) + cp.multiply(p3, costs[i, 2]) + cp.multiply(p4, costs[i, 3])

        # Budget constraint for the current patient
        #constraints.append(cp.sum(expected_costs) <= 20)  # Assuming a budget of 40 for each patient
        
        # Define the objective to maximize the net gain-to-cost ratio
        objective = cp.Maximize(expected_gains * gain_to_cost - expected_costs)  # Maximize the expected gain
        
        # Define and solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Store the optimal values for the current patient
        p1_values[i] = p1.value
        p2_values[i] = p2.value
        p3_values[i] = p3.value
        p4_values[i] = p4.value

    # Return the arrays of optimal p1, p2, p3, p4 for all patients
    return p1_values, p2_values, p3_values, p4_values
"""





"""

def optimize_p1234(gain_to_cost, gains, costs, B):

    num_patients = gains.shape[0]  # Assuming 'gains' is a matrix with shape (num_patients, 4)
    
    # Store results for all patients
    p1_values = np1.zeros(num_patients)
    p2_values = np1.zeros(num_patients)
    p3_values = np1.zeros(num_patients)
    p4_values = np1.zeros(num_patients)

    # Optimization variables for p1, p2, p3, p4 for each patient
    p1_vars = []
    p2_vars = []
    p3_vars = []
    p4_vars = []

    total_cost = 0  # Initialize total cost accumulator
    total_expected_gains = []
    

    constraints = []  # List to store all constraints

    # Loop through each patient (each row of the matrix)
    for i in range(num_patients):
        # Create optimization variables for the current patient
        p1 = cp.Variable()  # Probability for policy 00
        p2 = cp.Variable()  # Probability for policy 10
        p3 = cp.Variable()  # Probability for policy 01
        p4 = cp.Variable()  # Probability for policy 11
        
        # Append to lists for later reference
        p1_vars.append(p1)
        p2_vars.append(p2)
        p3_vars.append(p3)
        p4_vars.append(p4)

        # Constraints ensuring probabilities sum to 1 and are non-negative for each patient
        constraints += [
            p1 + p2 + p3 + p4 == 1,
            p1 >= 0, p2 >= 0, p3 >= 0, p4 >= 0
        ]
        
        # Expected gains and costs for the current patient
        expected_gains = cp.multiply(p1, gains[i, 0]) + cp.multiply(p2, gains[i, 1]) + cp.multiply(p3, gains[i, 2]) + cp.multiply(p4, gains[i, 3])
        expected_costs = cp.multiply(p1, costs[i, 0]) + cp.multiply(p2, costs[i, 1]) + cp.multiply(p3, costs[i, 2]) + cp.multiply(p4, costs[i, 3])
        total_expected_gains.append(expected_gains)
        # Add the expected costs to the total cost accumulator
        total_cost += expected_costs

    # Add the total cost constraint (applied after iterating over all patients)
    constraints.append(total_cost / num_patients <= B)

    for gain in total_expected_gains:
        objective = cp.Maximize(gain)

        problem = cp.Problem(objective, constraints)
        problem.solve()

    # Objective: maximize the total gain-to-cost ratio
    #objective = cp.Maximize(total_expected_gains)

    # Define and solve the optimization problem
    #problem = cp.Problem(objective, constraints)
    #problem.solve()

    # Extract the optimal values for each patient
    for i in range(num_patients):
        p1_values[i] = p1_vars[i].value
        p2_values[i] = p2_vars[i].value
        p3_values[i] = p3_vars[i].value
        p4_values[i] = p4_vars[i].value

    # Return the arrays of optimal p1, p2, p3, p4 for all patients
    return p1_values, p2_values, p3_values, p4_values
"""


def optimize_p1234(gain_to_cost, gains, costs, B):
    """
    Use convex optimization to determine the optimal p1, p2, p3, p4 for each patient,
    representing the probabilities for policies 00, 10, 01, 11.
    Solve the problem for each patient iteratively, while keeping track of the total cost.
    """
    num_patients = gains.shape[0]  # Assuming 'gains' is a matrix with shape (num_patients, 4)
    max_total_cost = B * num_patients

    # Store results for all patients
    p1_values = np1.zeros(num_patients)
    p2_values = np1.zeros(num_patients)
    p3_values = np1.zeros(num_patients)
    p4_values = np1.zeros(num_patients)

    total_cost = 0  # Initialize total cost accumulator

    # Loop through each patient (each row of the matrix)
    for i in range(num_patients):
        # Create optimization variables for the current patient
        p1 = cp.Variable(boolean=True)  # Probability for policy 00
        p2 = cp.Variable(boolean=True)  # Probability for policy 10
        p3 = cp.Variable(boolean=True)  # Probability for policy 01
        p4 = cp.Variable(boolean=True)  # Probability for policy 11

        # Constraints ensuring probabilities sum to 1 and are non-negative for the current patient
        constraints = [
            p1 + p2 + p3 + p4 == 1,
            p1 >= 0, p2 >= 0, p3 >= 0, p4 >= 0
        ]

        # Expected costs for the current patient
        expected_costs = (cp.multiply(p1, costs[i, 0]) + 
                          cp.multiply(p2, costs[i, 1]) + 
                          cp.multiply(p3, costs[i, 2]) + 
                          cp.multiply(p4, costs[i, 3]))

        # Add the expected costs to the total cost accumulator
        # Note: This should be checked to ensure it does not exceed the max total cost constraint
        #constraints.append(total_cost + expected_costs <= max_total_cost)
        constraints.append(expected_costs <= B)

        # Expected gains for the current patient
        expected_gains = (cp.multiply(p1, gains[i, 0]) + 
                          cp.multiply(p2, gains[i, 1]) + 
                          cp.multiply(p3, gains[i, 2]) + 
                          cp.multiply(p4, gains[i, 3]))

        # Define the objective to maximize for the current patient
        objective = cp.Maximize(expected_gains * gain_to_cost - expected_costs)
        #objective = cp.Maximize(expected_gains)
        
        # Define and solve the optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # Store the optimal values for the current patient
        p1_values[i] = p1.value
        p2_values[i] = p2.value
        p3_values[i] = p3.value
        p4_values[i] = p4.value

        # Update the total cost after solving for the current patient
        total_cost += expected_costs.value

        # Print or log the results for the current patient if needed
        #print(f"Patient {i}: p1 = {p1_values[i]}, p2 = {p2_values[i]}, p3 = {p3_values[i]}, p4 = {p4_values[i]}, Total Cost = {total_cost}")

    # Return the arrays of optimal p1, p2, p3, p4 for all patients
    return p1_values, p2_values, p3_values, p4_values

#optimize p1, p2, p3, p4 values
p1_optimal, p2_optimal, p3_optimal, p4_optimal = optimize_p1234(6.35, gains, costs, 20)

optimal_pmf = np1.array([p1_optimal, p2_optimal, p3_optimal, p4_optimal]).T
print("Optimal PMF for pi:", optimal_pmf)



def get_policy_from_p1234(p1_optimal, p2_optimal, p3_optimal, p4_optimal):
    """
    Determine the final policy for each patient using the optimal p1, p2, p3, and p4 values.
    """
    policy = []
    
    for i in range(xs.shape[0]):
        #construct PMF for the four possible policies using the optimal p1, p2, p3, and p4
        pmf = np1.array([p1_optimal[i], p2_optimal[i], p3_optimal[i], p4_optimal[i]])
        
        #select the policy with the highest probability
        optimal_policy = np1.argmax(pmf)
        policy.append(optimal_policy)

    return np1.array(policy)

# Get the final policy based on the optimal p1, p2, p3, and p4
policy = get_policy_from_p1234(p1_optimal, p2_optimal, p3_optimal, p4_optimal)
#policy = np.argmax(gains * 6.375 - costs, axis=-1)

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