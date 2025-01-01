import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import beta
from collections import Counter

def _simulate_singlekey_indep(x, key, params): 
    # ENVIRONMENT:

    key, subkey = jax.random.split(key)
    events_indep = jax.random.uniform(subkey, shape=(x.size,))
    events_indep = events_indep < x

    key, subkey = jax.random.split(key)
    events_time_indep = params['mean_event_times']  # Tn
    events_time_indep = events_time_indep + .1 * jax.random.normal(subkey, shape=(x.size,))
    events_time_indep = np.where(events_indep, events_time_indep, np.inf)

    key, subkey = jax.random.split(key)
    death_time_indep = params['mean_death_time']
    death_time_indep = death_time_indep + .1 * jax.random.normal(subkey)

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

    diagnoses_indep = np.any(posteriors_indep > params['diagnosis_threshold'], axis=-1)
    diagnoses_time_indep = np.where(diagnoses_indep, np.argmax(posteriors_indep > params['diagnosis_threshold'], axis=-1), np.inf)

    # GAIN-COST:
    events_time_diag_indep = np.where(diagnoses_time_indep < events_time_indep, np.inf, events_time_indep)  # T*

    survival_0_1_indep = np.minimum(death_time_indep, events_time_indep[0])
    survival_1_1_indep = np.minimum(death_time_indep, events_time_diag_indep[0])
    survival_0_2_indep = np.minimum(death_time_indep, events_time_indep[1])
    survival_1_2_indep = np.minimum(death_time_indep, events_time_diag_indep[1])

    samples_time_indep = np.minimum(diagnoses_time_indep, params['sample_sizes'].sum(axis=-1))

    tau_k_indep = params["sampling_freq"]
    ck_indep = params['screening_costs']
    
    """ cost_0_1_indep = 0
    cost_1_1_indep = ck_indep[0] * np.floor(survival_1_1_indep / tau_k_indep[0])
    cost_0_2_indep = 0
    cost_1_2_indep = ck_indep[1] * np.floor(survival_1_2_indep / tau_k_indep[0]) """
    
    cost_0_1_indep = 0
    cost_1_1_indep = ck_indep[0] * np.floor(np.minimum(survival_1_1_indep, samples_time_indep[0]))
    cost_0_2_indep = 0
    cost_1_2_indep = ck_indep[1] * np.floor(np.minimum(survival_1_2_indep, samples_time_indep[1]))

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

    diagnoses = np.any(posteriors > params['diagnosis_threshold'], axis=-1)
    diagnoses_time = np.where(diagnoses, np.argmax(posteriors > params['diagnosis_threshold'], axis=-1), np.inf)

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
    
    
    
    """ cost_00 = 0
    cost_10 = ck[0] * np.floor(survival_10 / tau_k[0])
    cost_01 = ck[1] * np.floor(survival_01 / tau_k[1])
    cost_11 = ck[0] * np.floor(survival_11 / tau_k[0]) + ck[1] * np.floor(survival_11 / tau_k[1]) """
    
    
    cost_00 = 0
    cost_10 = ck[0] * np.floor(np.minimum(survival_10, samples_time[0]))
    cost_01 = ck[1] * np.floor(np.minimum(survival_01, samples_time[1]))
    cost_11 = 0
    cost_11 = cost_11 + ck[0] * np.floor(np.minimum(survival_11, samples_time[0]))
    cost_11 = cost_11 + ck[1] * np.floor(np.minimum(survival_11, samples_time[1]))
   
    #gain_00, gain_10, gain_01, gain_11
    
    return (
        np.array([survival_00, survival_10, survival_01, survival_11]), 
        np.array([cost_00, cost_10, cost_01, cost_11]))

###

N_INNER = 200
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
params['mean_event_times'] = np.array([7., 7.])
params['sample_sizes'] = np.ones((1, 10))
# params['sample_perrors'] = np.array([.2, .2])
params['diagnosis_threshold'] = .95
params['screening_costs'] = np.array([1, 1]) #screening costs ck, will be modified accordingly.
params['sampling_freq'] = np.array([5., 5.]) #assume now as a sampling with fixed frequency. 

###



def monte_carlo_simulation_indep(num_iterations, xs, key, params, gamma_values, tau_k_values):
    """
    Perform Monte Carlo simulations by trying different values of gamma and tau_k's.
    Returns the expected gains and costs for each patient.
    """
    # Initialize accumulators for gains and costs
    N = xs.shape[0]
    total_gains = np.zeros((N, 4))  # For policies 00, 10, 01, 11
    total_costs = np.zeros((N, 4))
    
    for _ in range(num_iterations):
        # Sample gamma and tau_k's for this iteration
        key, subkey = jax.random.split(key)
        gamma = jax.random.choice(subkey, gamma_values)
        tau_k = jax.random.choice(subkey, tau_k_values, shape=(2,)) #k=2 is the nb of targets
        params['diagnosis_threshold'] =  gamma
        params['sampling_freq'] = tau_k
        #print(params)

        key, subkey = jax.random.split(key)
        gains, costs = simulate_batch_indep(xs, subkey, params)
        
        # Accumulate the results
        total_gains += gains
        total_costs += costs

    # Compute expected values by averaging over the iterations
    expected_gains = total_gains / num_iterations
    expected_costs = total_costs / num_iterations
    
    print("Expected gains:", expected_gains)
    print("Expected costs:", expected_costs)

    return expected_gains, expected_costs

def monte_carlo_simulation(num_iterations, xs, key, params, gamma_values, tau_k_values):
    """
    Perform Monte Carlo simulations by trying different values of gamma and tau_k's.
    Returns the expected gains and costs for each patient.
    """
    # Initialize accumulators for gains and costs
    N = xs.shape[0]
    total_gains = np.zeros((N, 4))  # For policies 00, 10, 01, 11
    total_costs = np.zeros((N, 4))
    
    for _ in range(num_iterations):
        # Sample gamma and tau_k's for this iteration
        key, subkey = jax.random.split(key)
        gamma = jax.random.choice(subkey, gamma_values)
        tau_k = jax.random.choice(subkey, tau_k_values, shape=(2,)) #k=2 is the nb of targets
        params['diagnosis_threshold'] =  gamma
        params['sampling_freq'] = tau_k
        #print(params)

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


num_iterations = 10  # Number of Monte Carlo iterations
gamma_values = np.linspace(0.8, 1.0, 20)  # Range of gamma values to try
tau_k_values = np.arange(1, 6, 1)  # Range of tau values to try

# Modify the code to systematically use all combinations of x1 and x2 from the grid
x1 = np1.linspace(0, 1, 100)
x2 = np1.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
xs = np.stack([X1.ravel(), X2.ravel()], axis=-1)  # Create grid combinations

""" # Ensure PMF matches grid
joint_pmf = np1.random.beta(3, 2, (100, 100))
joint_pmf /= joint_pmf.sum()
pmf_x = joint_pmf.ravel() """

# Parameters for the Beta distributions
alpha1, beta1 = 1, 1  # Parameters for X1
alpha2, beta2 = 1, 1  # Parameters for X2

# Define the grid
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
dx = x1[1] - x1[0]  # Interval width

# Compute the PDF values
pdf_x1 = beta.pdf(x1, alpha1, beta1)
pdf_x2 = beta.pdf(x2, alpha1, beta1)

# Compute the joint PMF as the outer product of the two PDFs
joint_pmf = np.outer(pdf_x1, pdf_x2) * dx * dx

# Normalize the PMF to ensure the sum equals 1
joint_pmf /= np.sum(joint_pmf)
pmf_x = joint_pmf.flatten()


# Run Monte Carlo simulation to get expected gains and costs
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
expected_gains_indep, expected_costs_indep = monte_carlo_simulation_indep(10, xs, key, params, gamma_values, tau_k_values)

# Extract the survival and costs for the first disease
survival_disease_1 = np1.array([expected_gains_indep[:, 0], expected_gains_indep[:, 1]])  # survival_0_1 and survival_1_1
cost_disease_1 = np1.array([expected_costs_indep[:,0], expected_costs_indep[:,1]])  # cost_0_1 and cost_1_1

# Extract the survival and costs for the second disease
survival_disease_2 = np1.array([expected_gains_indep[:,2], expected_gains_indep[:,3]])  # survival_0_2 and survival_1_2
cost_disease_2 = np1.array([expected_costs_indep[:,2], expected_costs_indep[:,3]]) # cost_0_2 and cost_1_2

expected_gains, expected_costs = monte_carlo_simulation(10, xs, key, params, gamma_values, tau_k_values)



def optimize_linear(x, pmf_x, expected_gains, expected_costs, k, B):
    N = x.shape[0]
    a = 2**k #nb of possible combinations for policies
    
    #optimization variables (q's)
    q = cp.Variable(a*N) #corresponds to the "x" variable in the linear CVXPY program
    
    gain_product = (expected_gains.T * pmf_x).T
    c = gain_product.flatten()
    print("Size of c=", c.shape)
    
    constraints = []
    constraints.append(q >= 0) #we can add the constraint that every q should be non-negative like this
    
    cost_product = (expected_costs.T * pmf_x).T
    A = cost_product.flatten()
    print("Size of A=", A.shape)
    
    constraints.append( A @ q <= B) #the cost constraint
    #I didn't apply the cost constraint and the sum of q(a|x) = 1 constraints together 
    # since one of them is an inequality and the other is an equality
    
    O = np1.zeros((N, a * N)) #to create the sum of q(a|x) = 1 constr
    for i in range(N):
        O[i, 4 * i : 4 * (i + 1)] = 1
        
    p = np1.ones(N)
    constraints.append( O @ q == p)
    
    prob = cp.Problem(cp.Maximize(c.T @ q), constraints)
    prob.solve()
    
    q_s = q.value
    optimal_qs = np1.zeros(N)
    
    #choosing the maximum probability for each feature x
    for i in range(N):
        optimal_qs[i] = np.argmax(q_s[4 * i : 4 * (i + 1)])
    
    return optimal_qs


def optimize_separately(x1, x2, pmf_x, expected_gains_x1, expected_costs_x1, expected_gains_x2, expected_costs_x2, k, B):
    N = x1.shape[0]
    a = 2 # number of possible combinations for policies
    
    #pmf = np1.ones(shape = pmf_x.shape)
    #pmf_x = pmf / 10000
    
    # Optimization variables (q's) for both x1 and x2
    q1 = cp.Variable(a * N)  # for disease 1 (x1)
    q2 = cp.Variable(a * N)  # for disease 2 (x2)

    gain_product_x1 = (expected_gains_x1 * pmf_x).T
    c1 = gain_product_x1.flatten()

    gain_product_x2 = (expected_gains_x2 * pmf_x).T
    c2 = gain_product_x2.flatten()

    constraints_1 = []
    constraints_2 = []
    constraints_1.append(q1 >= 0)  # non-negative q1
    constraints_2.append(q2 >= 0)  # non-negative q2

    cost_product_x1 = (expected_costs_x1 * pmf_x).T
    A1 = cost_product_x1.flatten()

    cost_product_x2 = (expected_costs_x2 * pmf_x).T
    A2 = cost_product_x2.flatten()

    constraints_1.append(A1 @ q1 <= B)  # cost constraint for disease 1
    constraints_2.append(A2 @ q2 <= B)  # cost constraint for disease 2

    O = np1.zeros((N, a * N))  # Sum of q(a|x) = 1 constraint for both diseases
    for i in range(N):
        O[i, a * i : a * (i + 1)] = 1
    
    p = np1.ones(N)
    constraints_1.append(O @ q1 == p)  # Constraint for disease 1
    constraints_2.append(O @ q2 == p)  # Constraint for disease 2
    
    # Solve the optimization problem
    prob1 = cp.Problem(cp.Maximize(c1.T @ q1), constraints_1)
    prob2 = cp.Problem(cp.Maximize(c2.T @ q2), constraints_2)

    prob1.solve()
    prob2.solve()

    q1_s = q1.value
    q2_s = q2.value
    
    # Get the optimal policies for each disease
    optimal_qs_x1 = np1.zeros(N)
    optimal_qs_x2 = np1.zeros(N)

    for i in range(N):
        optimal_qs_x1[i] = np.argmax(q1_s[a * i : a * (i + 1)])
        optimal_qs_x2[i] = np.argmax(q2_s[a * i : a * (i + 1)])

    return optimal_qs_x1, optimal_qs_x2
   

#get the final policy based on the q's
optimal_qs_x1, optimal_qs_x2 = optimize_separately(
    xs[:, 0], xs[:, 1], pmf_x, survival_disease_1, cost_disease_1, survival_disease_2, cost_disease_2, 2, 1.5)

#get the final policy based on the q's
policy = optimize_linear(xs, pmf_x, expected_gains, expected_costs, 2, 3)

print("Optimal qs for x1: ", optimal_qs_x1)
print("Optimal qs for x2: ", optimal_qs_x2)

def get_independent_policy(optimal_qs_x1, optimal_qs_x2):
    optimal_qs_x1 = optimal_qs_x1.astype(int)
    optimal_qs_x2 = optimal_qs_x2.astype(int)
    
    # Combine the independent decisions for x1 and x2 into a single policy
    independent_policy = (optimal_qs_x2 << 1) | optimal_qs_x1  # Bitwise operation to encode policy
    return independent_policy

# Derive the independent policy
independent_policy = get_independent_policy(optimal_qs_x1, optimal_qs_x2)

def calculate_policy_differences(policy, independent_policy):
    # Count occurrences of each policy (0, 1, 2, 3) in both `policy` and `independent_policy`
    policy_counts = Counter(policy)
    independent_policy_counts = Counter(independent_policy)
    
    # Ensure all policies (0, 1, 2, 3) are included in the counts
    for i in range(4):
        policy_counts[i] = policy_counts.get(i, 0)
        independent_policy_counts[i] = independent_policy_counts.get(i, 0)

    # Calculate the differences for specified policies
    diff_1_3 = (policy_counts[1] + policy_counts[3]) - (independent_policy_counts[1] + independent_policy_counts[3])
    diff_2_3 = (policy_counts[2] + policy_counts[3]) - (independent_policy_counts[2] + independent_policy_counts[3])
    diff_1_2_3 = (policy_counts[1] + policy_counts[2] + policy_counts[3]) - (
        independent_policy_counts[1] + independent_policy_counts[2] + independent_policy_counts[3])

    return {
        "policy_counts": dict(policy_counts),
        "independent_policy_counts": dict(independent_policy_counts),
        "diff_1_3": diff_1_3,
        "diff_2_3": diff_2_3,
        "diff_1_2_3": diff_1_2_3
    }
    
results = calculate_policy_differences(policy, independent_policy)

# Print the results
print("Policy counts:", results["policy_counts"])
print("Independent policy counts:", results["independent_policy_counts"])
print("Difference (1 & 3):", results["diff_1_3"])
print("Difference (2 & 3):", results["diff_2_3"])
print("Difference (1, 2, & 3):", results["diff_1_2_3"])


# Extract survival times for the chosen policy
def get_survival_times(xs, policy, expected_gains):
    survival_times = np1.zeros_like(policy, dtype=float)
    for i in range(xs.shape[0]):
        chosen_policy = int(policy[i])
        survival_times[i] = expected_gains[i, chosen_policy]
    return survival_times

survival_times = get_survival_times(xs, policy, expected_gains)

def plot_policy_with_survival(xs, policy, survival_times):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(xs[:, 0], xs[:, 1], c=survival_times, cmap='viridis', s=50, label='Chosen Policy')
    cbar = plt.colorbar(sc)
    #plt.title('Survival Times under Our Model')
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    

    plt.show()

plot_policy_with_survival(xs, policy, survival_times)

#plot the resulting policy assignments
def plot_policy_separately(optimal_qs_x1, optimal_qs_x2, xs):
    plt.figure(figsize=(6, 6))

    # Plot for the four combinations of disease 1 (x1)
    plt.scatter(xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 0), 0], xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 0), 1], color='silver', s=50, label='a = 1 (00)')
    plt.scatter(xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 0), 0], xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 0), 1], color='tab:pink', s=50, label='a = 2 (10)')
    plt.scatter(xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 1), 0], xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 1), 1], color='tab:purple', s=50, label='a = 3 (01)')
    plt.scatter(xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 1), 0], xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 1), 1], color='tab:red', s=50, label='a = 4 (11)')

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend()
    #plt.title('Independent Screening for Two Diseases')
    plt.show()
    
    
    plt.figure(figsize=(6, 6))

    # Plot for the four combinations of disease 1 (x1)
    plt.scatter(xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 0), 0], xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 0), 1], color='silver', s=50, label='a = 1 (00)')
    plt.scatter(xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 0), 0], xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 0), 1], color='tab:pink', s=50, label='a = 2 (10)')
    plt.scatter(xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 1), 0], xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 1), 1], color='tab:purple', s=50, label='a = 3 (01)')
    plt.scatter(xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 1), 0], xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 1), 1], color='tab:red', s=50, label='a = 4 (11)')

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.legend()
    plt.title('Independent Screening for Two Diseases')
    plt.show()
    
#plot the optimized policy results
plot_policy_separately(optimal_qs_x1, optimal_qs_x2, xs)

def plot_policy_differences(xs, our_policy, independent_policy, save_path='figs/policy_differences_reversed.pdf'):
    """
    Plots the differences between the optimized policies from our joint model and the independent policies.

    Parameters:
    - xs: np.ndarray, shape (n_samples, 2)
      The feature grid (x1, x2) values.
    - our_policy: np.ndarray, shape (n_samples,)
      Policies optimized using our joint model.
    - independent_policy: np.ndarray, shape (n_samples,)
      Policies determined independently for each disease.
    - save_path: str
      Path to save the resulting figure.

    Returns:
    - None
    """
    # Define colors for each mismatch type
    colors = ['silver', 'tab:pink', 'tab:purple', 'tab:red']
    labels = ['00 -> other', '01 -> other', '10 -> other', '11 -> other']

    plt.figure(figsize=(8, 6))

    # Loop through the four possible policy assignments
    for policy_value in range(4):
        mask = (independent_policy == policy_value) & (our_policy != policy_value)
        plt.scatter(
            xs[mask, 0], xs[mask, 1],
            color=colors[policy_value],
            s=50, label=f'a = {policy_value + 1}'
        )

    # Configure the plot
    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.title('Policy Differences: Our Model vs. Independent')
    plt.legend()
    #plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    plt.figure(figsize=(8, 6))

    # Loop through the four possible policy assignments
    for policy_value in range(4):
        mask = (independent_policy == policy_value) & (our_policy != policy_value)
        plt.scatter(
            xs[mask, 0], xs[mask, 1],
            color=colors[policy_value],
            s=50, label=f'a = {policy_value + 1}'
        )

    # Configure the plot
    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.title('Policy Differences: Our Model vs. Independent')
    plt.legend()
    #plt.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

plot_policy_differences(xs, independent_policy, policy)


# Compute survival times based on optimal policies
survival_time_x1 = np1.zeros(xs.shape[0])
survival_time_x2 = np1.zeros(xs.shape[0])

for i in range(xs.shape[0]):
    # Retrieve optimal policy indices
    policy_x1 = int(optimal_qs_x1[i])
    policy_x2 = int(optimal_qs_x2[i])
    
    # Use policy indices to select survival gains for each disease
    survival_time_x1[i] = survival_disease_1[policy_x1, i]
    survival_time_x2[i] = survival_disease_2[policy_x2, i]

# Reshape survival times back to the grid shape
survival_time_x1_grid = survival_time_x1.reshape(X1.shape)
survival_time_x2_grid = survival_time_x2.reshape(X2.shape)

# Plot heatmaps for each disease
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Disease 1 Survival
im1 = axes[0].imshow(
    survival_time_x1_grid, extent=(0, 1, 0, 1), origin='lower', aspect='auto', cmap='viridis'
)
axes[0].set_title("Survival Time for Disease 1")
axes[0].set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"$x_2$")
fig.colorbar(im1, ax=axes[0])

# Disease 2 Survival
im2 = axes[1].imshow(
    survival_time_x2_grid, extent=(0, 1, 0, 1), origin='lower', aspect='auto', cmap='viridis'
)
axes[1].set_title("Survival Time for Disease 2")
axes[1].set_xlabel(r"$x_1$")
axes[1].set_ylabel(r"$x_2$")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# Plot heatmaps for each disease
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Disease 1 Survival
im1 = axes[0].imshow(
    survival_time_x1_grid, extent=(0, 1, 0, 1), origin='lower', aspect='auto', cmap='viridis'
)
#axes[0].set_title("Survival Time for Disease 1")
axes[0].set_xlabel(r"$x_1$")
axes[0].set_ylabel(r"$x_2$")
fig.colorbar(im1, ax=axes[0])

# Disease 2 Survival
im2 = axes[1].imshow(
    survival_time_x2_grid, extent=(0, 1, 0, 1), origin='lower', aspect='auto', cmap='viridis'
)
#axes[1].set_title("Survival Time for Disease 2")
axes[1].set_xlabel(r"$x_1$")
axes[1].set_ylabel(r"$x_2$")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

min_survival_time_indep = np.minimum(survival_time_x1, survival_time_x2)

# Reshape minimum survival times back to the grid shape
min_survival_time_grid = min_survival_time_indep.reshape(X1.shape)

plt.figure(figsize=(8, 6))
im = plt.imshow(
    min_survival_time_grid, extent=(0, 1, 0, 1), origin="lower", aspect="auto", cmap="viridis"
)
#plt.title("Minimum Survival Time Between Two Diseases for Independent Screening")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar(im)
#cbar.set_label("Survival Time")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
im = plt.imshow(
    min_survival_time_grid, extent=(0, 1, 0, 1), origin="lower", aspect="auto", cmap="viridis"
)
plt.title("Minimum Survival Time Between Two Diseases for Independent Screening")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar(im)
#cbar.set_label("Survival Time")

plt.tight_layout()
plt.show()


difference = survival_times - min_survival_time_indep
print("Difference in survival times:", difference.mean())
mean_diff = difference.mean()

# Apply a mask to filter values greater than zero
mask = difference > 0
filtered_xs = xs[mask]
filtered_difference = difference[mask]

# Plot the heatmap for the filtered survival time differences
plt.figure(figsize=(8, 6))
sc = plt.scatter(filtered_xs[:, 0], filtered_xs[:, 1], c=filtered_difference, cmap='viridis', s=50, label='Chosen Policy')
#plt.title("Filtered Survival Time Difference (Our Model - Independent)")
plt.xlim(0, 1)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar(sc)
#cbar.set_label("Survival Time")

plt.tight_layout()
plt.show()


# Plot the heatmap for the filtered survival time differences
plt.figure(figsize=(8, 6))
sc = plt.scatter(filtered_xs[:, 0], filtered_xs[:, 1], c=filtered_difference, cmap='viridis', s=50, label='Chosen Policy')
plt.title("Filtered Survival Time Difference (Our Model - Independent)")
plt.xlim(0, 1)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar(sc)
#cbar.set_label("Survival Time")

plt.tight_layout()
plt.show()


# Plot the heatmap for the filtered survival time differences
plt.figure(figsize=(8, 6))
sc = plt.scatter(xs[:, 0], xs[:, 1], c=difference, cmap='viridis', s=50, label='Chosen Policy')
#plt.title(f"Survival Time Difference (Our Model - Independent), Mean = {mean_diff:.3f}")
plt.xlim(0, 1)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar(sc)
#cbar.set_label("Survival Time")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sc = plt.scatter(xs[:, 0], xs[:, 1], c=difference, cmap='viridis', s=50, label='Chosen Policy')
plt.title(f"Survival Time Difference (Our Model - Independent)")
plt.xlim(0, 1)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar(sc)
#cbar.set_label("Survival Time")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sc = plt.scatter(xs[:, 0], xs[:, 1], c=difference, cmap='viridis', s=50, label='Chosen Policy')
plt.title(f"Survival Time Difference (Our Model - Independent), Mean = {mean_diff:.3f}")
plt.xlim(0, 1)  # Set x-axis limits
plt.ylim(0, 1)  # Set y-axis limits
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
cbar = plt.colorbar(sc)
#cbar.set_label("Survival Time")

plt.tight_layout()
plt.show()