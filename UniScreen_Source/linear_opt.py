import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import beta

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
    cost_11 = ck[0] * np.floor(survival_11 / tau_k[0]) + ck[1] * np.floor(survival_11 / tau_k[1])
    """

    
    
    cost_00 = 0
    cost_10 = ck[0] * np.floor(np.minimum(survival_10, samples_time[0]))
    cost_01 = ck[1] * np.floor(np.minimum(survival_01, samples_time[1]))
    cost_11 = 0
    cost_11 = cost_11 + ck[0] * np.floor(np.minimum(survival_11, samples_time[0]))
    cost_11 = cost_11 + ck[1] * np.floor(np.minimum(survival_11, samples_time[1]))
    
    
    """ cost_00 = 0
    cost_10 = ck[0] * np.floor(survival_10 / tau_k[0])
    cost_01 = ck[1] * np.floor(survival_01 / tau_k[1])
    cost_11 = ck[0] * np.floor(survival_11 / tau_k[0]) + ck[1] * np.floor(survival_11 / tau_k[1])
    """
    
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
params['sample_sizes'] = np.ones((2, 10))
# params['sample_perrors'] = np.array([.2, .2])
params['diagnosis_threshold'] = .95
params['screening_costs'] = np.array([1, 1]) #screening costs ck, will be modified accordingly.
params['sampling_freq'] = np.array([5., 5.]) #assume now as a sampling with fixed frequency. 

###

# Monte Carlo Process
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
        tau_k = jax.random.choice(subkey, tau_k_values) #k=2 is the nb of targets
        params['diagnosis_threshold'] =  gamma
        params['sampling_freq'] = (tau_k, tau_k)
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


def sample_from_pmf(pmf, key, N):
    pmf_flat = pmf.flatten()
    idx = jax.random.choice(key, np.arange(pmf.size), p=pmf_flat, shape=(N,)) #make a random choice, given the sample number N
    x1_idx = idx // pmf.shape[1] #determine the row/column
    x2_idx = idx % pmf.shape[1]
    x1_vals = x1[x1_idx]
    x2_vals = x2[x2_idx]
    xs = np.stack([x1_vals, x2_vals], axis=-1)
    pmf_xs = pmf_flat[idx]
    return xs, pmf_xs


num_iterations = 10  # Number of Monte Carlo iterations
gamma_values = np.linspace(0.8, 1.0, 20)  # Range of gamma values to try
tau_k_values = np.arange(1, 6, 1)  # Range of tau values to try


#create PMF on 100x100 grid now
x1 = np1.linspace(0, 1, 100)
x2 = np1.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
xs = np.stack([X1.ravel(), X2.ravel()], axis=-1)  # Create grid combinations

# Ensure PMF matches grid alpha =3 beta =2
""" alpha = 7
b = 7
joint_pmf = np1.random.beta(alpha, b, (100, 100))
joint_pmf /= joint_pmf.sum()
pmf_x = joint_pmf.ravel() """


#joint_pmf = np1.random.uniform(1, 6, size=(100, 100))
"""
alpha = 3
beta = 2
joint_pmf = np1.random.beta(alpha, beta, (100, 100))
joint_pmf /= joint_pmf.sum()
"""

""" # Step 3: Define the parameters of the Beta distributions
alpha1, beta1 = 2, 10  # for x1
alpha2, beta2 = 5, 1  # for x2

# Step 4: Compute the Beta PDFs for x1 and x2
pdf_x1 = beta.pdf(X1, alpha1, beta1)
pdf_x2 = beta.pdf(X2, alpha1, beta1)

# Step 5: Compute the joint PDF assuming independence
joint_pmf = pdf_x1 * pdf_x2
joint_pmf /= joint_pmf.sum()
 """


# Parameters for the Beta distributions
alpha1, beta1 = 5, 5  # Parameters for X1
alpha2, beta2 = 5, 5  # Parameters for X2

# Define the grid
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
dx = x1[1] - x1[0]  # Interval width

# Compute the PDF values
pdf_x1 = beta.pdf(x1, alpha1, beta1)
pdf_x2 = beta.pdf(x2, alpha2, beta2)

# Compute the joint PMF as the outer product of the two PDFs
joint_pmf = np.outer(pdf_x1, pdf_x2) * dx * dx

# Normalize the PMF to ensure the sum equals 1
joint_pmf /= np.sum(joint_pmf)

plt.imshow(joint_pmf, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='Probability')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Joint PMF of $X_1$ and $X_2$')
plt.show()

pmf_x = joint_pmf.flatten()

""" key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
_, pmf_x = sample_from_pmf(joint_pmf, subkey, N) """

"""
key, subkey = jax.random.split(key)
xs = jax.random.beta(subkey, a = 1, b = 1, shape=(N, 2))
"""


key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
expected_gains, expected_costs = monte_carlo_simulation(num_iterations, xs, key, params, gamma_values, tau_k_values)
#gains, costs = simulate_batch(xs, subkey, params)


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
    

#get the final policy based on the q's
policy = optimize_linear(xs, pmf_x, expected_gains, expected_costs, 2, 3.5)

#plot the resulting policy assignments
def plot_policy(policy):
    plt.figure()
    plt.title("Unified Screening for Two Diseases")
    plt.scatter(xs[policy == 0, 0], xs[policy == 0, 1], color='silver', s=50, label='a = 1 (00)')
    plt.scatter(xs[policy == 1, 0], xs[policy == 1, 1], color='tab:pink', s=50, label='a = 2 (10)')
    plt.scatter(xs[policy == 2, 0], xs[policy == 2, 1], color='tab:purple', s=50, label='a = 3 (01)')
    plt.scatter(xs[policy == 3, 0], xs[policy == 3, 1], color='tab:red', s=50, label='a = 4 (11)')

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.title("Unified Screening for Two Diseases")
    plt.scatter(xs[policy == 0, 0], xs[policy == 0, 1], color='silver', s=50, label='a = 1 (00)')
    plt.scatter(xs[policy == 1, 0], xs[policy == 1, 1], color='tab:pink', s=50, label='a = 2 (10)')
    plt.scatter(xs[policy == 2, 0], xs[policy == 2, 1], color='tab:purple', s=50, label='a = 3 (01)')
    plt.scatter(xs[policy == 3, 0], xs[policy == 3, 1], color='tab:red', s=50, label='a = 4 (11)')

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    #plt.title("Unified Screening for Two Diseases")
    plt.scatter(xs[policy == 0, 0], xs[policy == 0, 1], color='silver', s=50, label='a = 1 (00)')
    plt.scatter(xs[policy == 1, 0], xs[policy == 1, 1], color='tab:pink', s=50, label='a = 2 (10)')
    plt.scatter(xs[policy == 2, 0], xs[policy == 2, 1], color='tab:purple', s=50, label='a = 3 (01)')
    plt.scatter(xs[policy == 3, 0], xs[policy == 3, 1], color='tab:red', s=50, label='a = 4 (11)')

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.figure()
    #plt.title("Unified Screening for Two Diseases")
    plt.scatter(xs[policy == 0, 0], xs[policy == 0, 1], color='silver', s=50, label='a = 1 (00)')
    plt.scatter(xs[policy == 1, 0], xs[policy == 1, 1], color='tab:pink', s=50, label='a = 2 (10)')
    plt.scatter(xs[policy == 2, 0], xs[policy == 2, 1], color='tab:purple', s=50, label='a = 3 (01)')
    plt.scatter(xs[policy == 3, 0], xs[policy == 3, 1], color='tab:red', s=50, label='a = 4 (11)')

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    

#plot the optimized policy results
plot_policy(policy)


"""
policies = ['00', '10', '01', '11']
for i, policy in enumerate(policies):
    plt.hist(expected_gains[:, i], bins=30, alpha=0.5, label=f'Policy {policy}')
plt.xlabel("Survival Time")
plt.ylabel("Frequency")
plt.title("Survival Times for Different Policies")
plt.legend()
plt.show()


"""
