import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import beta
import matplotlib.pyplot as plt
import jax.numpy as jnp

def _simulate_singlekey(x, key, params):

    # ENVIRONMENT:

    key, subkey = jax.random.split(key)
    events = jax.random.uniform(subkey, shape=(x.size,))
    #events = np.round(events)
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
    cost_10 = np.floor(np.minimum(survival_10, samples_time[0]))
    cost_01 = np.floor(np.minimum(survival_01, samples_time[1]))
    cost_11 = 0
    cost_11 = cost_11 + np.floor(np.minimum(survival_11, samples_time[0]))
    cost_11 = cost_11 + np.floor(np.minimum(survival_11, samples_time[1]))
    
    #gain_00, gain_10, gain_01, gain_11
    #survival_00, survival_10, survival_01, survival_11
    
    return (
        np.array([survival_00, survival_10, survival_01, survival_11]), 
        np.array([cost_00, cost_10, cost_01, cost_11]))

###

N_INNER = 200
N = 500

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
params['mean_event_times'] = np.array([7.4, 7.4])
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

"""
joint_pmf = np1.ones((100, 100))
joint_pmf[:50, :50] = 2
joint_pmf[50:, 50:] = 2
joint_pmf /= joint_pmf.sum()
"""


#joint_pmf = np1.random.uniform(1, 6, size=(100, 100))
alpha = 3
beta = 2
joint_pmf = np1.random.beta(alpha, beta, (100, 100))
joint_pmf /= joint_pmf.sum()


"""
# Step 2: Create a grid of (x1, x2) pairs
X1, X2 = np.meshgrid(x1, x2)

# Step 3: Define the parameters of the Beta distributions
alpha1, beta1 = 3.0, 2.0  # for x1
alpha2, beta2 = 3.0, 2.0  # for x2

# Step 4: Compute the Beta PDFs for x1 and x2
pdf_x1 = beta.pdf(X1, alpha1, beta1)
pdf_x2 = beta.pdf(X2, alpha2, beta2)

# Step 5: Compute the joint PDF assuming independence
joint_pmf = pdf_x1 * pdf_x2
joint_pmf /= joint_pmf.sum()

"""



""" key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
_, pmf_x = sample_from_pmf(joint_pmf, subkey, N)

key, subkey = jax.random.split(key)
_ = jax.random.beta(subkey, a = 1, b = 1, shape=(N, 2))
#xs = xs.at[:, 1].set(0.5)

x1_values = np.linspace(0, 1, N)  # Equally spaced values for x1 from 0 to 1
x2_values = np.full_like(x1_values, 0.5)   # Second column set to 0.5

#Combine x1 and x2 into xs
xs = np.column_stack((x1_values, x2_values)) """

#create PMF on 100x100 grid now
x1 = np1.linspace(0, 1, 100)
x2 = np1.linspace(0, 1, 100)
X1, X2 = np.meshgrid(x1, x2)
xs = np.stack([X1.ravel(), X2.ravel()], axis=-1)  # Create grid combinations

# Ensure PMF matches grid alpha =3 beta =2
alpha = 7
b = 7
joint_pmf = np1.random.beta(alpha, b, (100, 100))
joint_pmf /= joint_pmf.sum()
pmf_x = joint_pmf.ravel()

key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
expected_gains, expected_costs = monte_carlo_simulation(num_iterations, xs, key, params, gamma_values, tau_k_values)
#gains, costs = simulate_batch(xs, subkey, params)


def plot_expected_gains_and_costs(xs, expected_gains, expected_costs):
    # Reshape the xs into individual x1 and x2 for plotting
    x1_vals = xs[:, 0]
    x2_vals = xs[:, 1]
    
    # Plot for each policy
    policies = ['00', '10', '01', '11']
    
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(4):
        # Plot expected gains for policy i
        sc1 = axs[0, i].scatter(x1_vals, x2_vals, c=expected_gains[:, i], cmap='coolwarm', s=10)
        axs[0, i].set_title(f"Expected Survival Time - Policy {policies[i]} (a= {i+1})")
        axs[0, i].set_xlabel(r"$x_1$")
        axs[0, i].set_ylabel(r"$x_2$")
        plt.colorbar(sc1, ax=axs[0, i])
        
        # Plot expected costs for policy i
        sc2 = axs[1, i].scatter(x1_vals, x2_vals, c=expected_costs[:, i], cmap='viridis', s=10)
        axs[1, i].set_title(f"Expected Cost - Policy {policies[i]} (a= {i+1})")
        axs[1, i].set_xlabel(r"$x_1$")
        axs[1, i].set_ylabel(r"$x_2$")
        plt.colorbar(sc2, ax=axs[1, i])

    plt.tight_layout()
    plt.show()
    
    for i in range(4):
        plt.figure()
        sc = plt.scatter(x1_vals, x2_vals, c=expected_gains[:, i], cmap='coolwarm', s=10)
        plt.title(f"Survival Time - λ * Cost: Policy {policies[i]}")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        #plt.colorbar(sc, label="Expected Survival Time")
        plt.colorbar(sc)
        plt.show()
        
        plt.figure()
        sc = plt.scatter(x1_vals, x2_vals, c=expected_costs[:, i], cmap='viridis', s=10)
        plt.title(f"Survival Time - λ * Cost: Policy {policies[i]}")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        #plt.colorbar(sc, label="Expected Survival Time")
        plt.colorbar(sc)
        plt.show()
    
    


def plot_gains_minus_costs(xs, expected_gains, expected_costs, lambda_val):
    # Reshape the xs into individual x1 and x2 for plotting
    x1_vals = xs[:, 0]
    x2_vals = xs[:, 1]
    
    # Calculate gain - lambda * cost for each policy
    gain_minus_cost = expected_gains - lambda_val * expected_costs
    
    # Policies labels
    policies = ['00', '10', '01', '11']
    
    # Create subplots for each policy
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(4):
        # Plot gain - lambda * cost for policy i
        sc = axs[i].scatter(x1_vals, x2_vals, c=gain_minus_cost[:, i], cmap='coolwarm', s=10)
        axs[i].set_title(f"Gain - λ * Cost: Policy {policies[i]} (a={i+1})")
        axs[i].set_xlabel(r"$x_1$")
        axs[i].set_ylabel(r"$x_2$")
        plt.colorbar(sc, ax=axs[i])

    plt.tight_layout()
    plt.show()
    
    

# Example usage:
lambda_val = 0.2  # Set the value of lambda
#plot_gains_minus_costs(xs, expected_gains, expected_costs, lambda_val)

def plot_gains_costs_with_decision_boundary(xs, expected_gains, expected_costs, lambda_val):
    # Reshape the xs into individual x1 and x2 for plotting
    x1_vals = xs[:, 0]
    x2_vals = xs[:, 1]
    
    # Calculate gain - lambda * cost for each policy
    gain_minus_cost = expected_gains - lambda_val * expected_costs
    
    # Find the maximum value across the 4 policies for each (x1, x2)
    max_gain_minus_cost = np.max(gain_minus_cost, axis=1)
    
    # Find the index (policy) that gives the maximum gain - lambda * cost for each (x1, x2)
    best_policy = np.argmax(gain_minus_cost, axis=1)
    best_policy += 1
    
    # Policies labels
    policies = ['00', '10', '01', '11']
    
    # Create subplots for each policy and the decision boundary
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(4):
        # Plot gain - lambda * cost for policy i
        sc = axs[0, i].scatter(x1_vals, x2_vals, c=gain_minus_cost[:, i], cmap='magma', s=10)
        axs[0, i].set_title(f"Survival Time - λ * Cost: Policy {policies[i]}")
        axs[0, i].set_xlabel(r"$x_1$")
        axs[0, i].set_ylabel(r"$x_2$")
        plt.colorbar(sc, ax=axs[0, i])

    # Plot the maximum value across all policies
    sc_max = axs[1, 1].scatter(x1_vals, x2_vals, c=max_gain_minus_cost, cmap='plasma', s=10)
    axs[1, 1].set_title("Maximum Survival Time - λ * Cost")
    axs[1, 1].set_xlabel(r"$x_1$")
    axs[1, 1].set_ylabel(r"$x_2$")
    plt.colorbar(sc_max, ax=axs[1, 1])

    # Plot the decision boundary, indicating which policy is optimal
    sc_boundary = axs[1, 2].scatter(x1_vals, x2_vals, c=best_policy, cmap='tab10', s=10)
    axs[1, 2].set_title("Decision Boundary (Best Policy)")
    axs[1, 2].set_xlabel(r"$x_1$")
    axs[1, 2].set_ylabel(r"$x_2$")
    plt.colorbar(sc_boundary, ax=axs[1, 2], ticks=range(4))
    
    # Hide the other unused subplot in the second row
    axs[1, 0].axis('off')
    axs[1, 3].axis('off')

    plt.tight_layout()
    plt.show()
    
    # Plot gain - lambda * cost for each policy on separate figures
    for i in range(4):
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(x1_vals, x2_vals, c=gain_minus_cost[:, i], cmap='magma', s=10)
        plt.title(f"Survival Time - λ * Cost: Policy {policies[i]}")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        #plt.colorbar(sc, label="Survival Time - λ * Cost")
        plt.colorbar(sc)
        plt.show()
    
    # Plot the maximum value across all policies in a separate figure
    plt.figure(figsize=(6, 5))
    sc_max = plt.scatter(x1_vals, x2_vals, c=max_gain_minus_cost, cmap='plasma', s=10)
    plt.title("Maximum Survival Time - λ * Cost")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    #plt.colorbar(sc_max, label="Max Survival Time - λ * Cost")
    plt.colorbar(sc_max)
    plt.show()
    
    plt.figure(figsize=(6, 5))
    sc_boundary = plt.scatter(x1_vals, x2_vals, c=best_policy, cmap='tab10')
    plt.title("Decision Boundary (Best Policy)")
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.colorbar(sc_boundary, ticks=range(4), label="Action Index")
    plt.show()

# Example usage:
lambda_val = 0.2  # Set the value of lambda
plot_gains_costs_with_decision_boundary(xs, expected_gains, expected_costs, lambda_val)



# Call the plotting function
plot_expected_gains_and_costs(xs, expected_gains, expected_costs)

def plot_x1_vs_gain_cost_for_policies(xs, expected_gains, expected_costs, lambda_val):
    x1_values = xs[:, 0]  # Extract x1 values
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots
    policies = ['Policy 00', 'Policy 10', 'Policy 01', 'Policy 11']
    
    for i in range(4):
        #gain_minus_cost = expected_gains[:, i] - lambda_val * expected_costs[:, i]  # Gain - Cost for each policy
        gain_minus_cost = expected_gains[:, i]
        row = i // 2  # Determine the row in the 2x2 grid
        col = i % 2   # Determine the column in the 2x2 grid
        
        ax[row, col].scatter(x1_values, gain_minus_cost, c='blue', alpha=0.5)
        ax[row, col].set_xlabel(r"$x_1$")
        ax[row, col].set_ylabel('Expected Survival Time')
        ax[row, col].set_title(policies[i])
        ax[row, col].grid(True)
        
        if policies[i] == 'Policy 10':
            ax[row, col].set_ylim(8.7, 9.3)
        
        if policies[i] == 'Policy 11':
            ax[row, col].set_ylim(9.9, 10.1)
    
    plt.tight_layout()
    plt.show()
    
    for i in range(4):
        # Calculate gain - lambda * cost for each policy
        gain_minus_cost = expected_gains[:, i]
        
        # Create a separate figure for each policy
        plt.figure(figsize=(6, 5))
        plt.scatter(x1_values, gain_minus_cost, c='blue', alpha=0.5)
        plt.xlabel(r"$x_1$")
        plt.ylabel('Expected Survival Time')
        plt.title(policies[i])
        plt.grid(True)
        
        # Set specific y-axis limits for Policy 10 and Policy 11
        if policies[i] == 'Policy 10':
            plt.ylim(8.6, 9.4)
        elif policies[i] == 'Policy 11':
            plt.ylim(9.9, 10.1)
        
        plt.show()

# Use the modified function to plot for all four policies
#plot_x1_vs_gain_cost_for_policies(xs, expected_gains, expected_costs, lambda_val)
