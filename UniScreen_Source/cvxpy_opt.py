import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Load the data
xs = np.load("res/xs.npy")
gains = np.load("res/gains.npy")
costs = np.load("res/costs.npy")

def optimize_pq(gain_to_cost):
    """
    Use convex optimization to determine the optimal p(x1, x2) and q(x1, x2) that maximize the gain.
    """
    num_patients = xs.shape[0]
    
    # Create optimization variables for p and q for each patient
    
    p = cp.Variable(num_patients)
    q = cp.Variable(num_patients)
    
    # Define constraints: p and q must be between 0 and 1 (valid probabilities)
    constraints = [
        p >= 0, p <= 1, 
        q >= 0, q <= 1
    ]
    
    # Compute the expected gain and cost for each patient based on p and q
    expected_gains = (1 - p) * (1 - q) * gains[:, 0] + p * (1 - q) * gains[:, 1] + (1 - p) * q * gains[:, 2] + p * q * gains[:, 3]
    expected_costs = (1 - p) * (1 - q) * costs[:, 0] + p * (1 - q) * costs[:, 1] + (1 - p) * q * costs[:, 2] + p * q * costs[:, 3]
    
    # Define the objective to maximize the net gain-to-cost ratio
    objective = cp.Maximize(cp.sum(expected_gains * gain_to_cost - expected_costs))
    
    # Solve the optimization problem
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the optimal values of p and q
    return p.value, q.value

# Optimize the p and q values
p_optimal, q_optimal = optimize_pq(gain_to_cost=6.375)

# Now, compute the final policy for each patient based on the optimal p and q
def get_policy_from_pq(p_optimal, q_optimal):
    """
    Determine the final policy for each patient using the optimal p and q values.
    """
    policy = []
    
    for i in range(xs.shape[0]):
        # Get the PMF for the four possible policies using the optimal p and q
        pmf = np.zeros(4)
        pmf[0] = (1 - p_optimal[i]) * (1 - q_optimal[i])  # Probability of policy 00
        pmf[1] = p_optimal[i] * (1 - q_optimal[i])        # Probability of policy 10
        pmf[2] = (1 - p_optimal[i]) * q_optimal[i]        # Probability of policy 01
        pmf[3] = p_optimal[i] * q_optimal[i]              # Probability of policy 11
        
        # Select the policy with the highest probability
        optimal_policy = np.argmax(pmf)
        policy.append(optimal_policy)

    return np.array(policy)

# Get the final policy based on the optimal p and q
policy = get_policy_from_pq(p_optimal, q_optimal)

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
