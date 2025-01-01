import numpy as np
from scipy.optimize import minimize

xs = np.load("res/xs.npy")
gains = np.load("res/gains.npy")
costs = np.load("res/costs.npy")

# Coefficients
c1 = gains[:, 0] 
c2 = gains[:, 1]  
c3 = gains[:, 2]  
c4 = gains[:, 3]  

num_scenarios = len(c1)

# Arrays to store optimal p and q for each scenario
optimal_p = np.zeros(num_scenarios)
optimal_q = np.zeros(num_scenarios)
optimal_values = np.zeros(num_scenarios)
optimal_pmf = np.zeros((num_scenarios, 4))

# Objective function (negated for maximization)
def objective(x, c1, c2, c3, c4):
    p, q = x
    # Calculate the expected value based on the current coefficients
    value = (1 - p) * (1 - q) * c1 + p * (1 - q) * c2 + q * (1 - p) * c3 + p * q * c4
    return -value  # Negate for maximization

# Loop over each scenario
for i in range(num_scenarios):
    # Initial guess for p and q (both as vectors)
    initial_guess = [0.5, 0.5]
    
    # Bounds for p and q (both between 0 and 1)
    bounds = [(0, 1), (0, 1)]
    
    # Minimize the negated objective function to achieve maximization for each scenario
    result = minimize(objective, initial_guess, args=(c1[i], c2[i], c3[i], c4[i]), bounds=bounds, method='L-BFGS-B')
    
    # Store the results
    optimal_p[i], optimal_q[i] = result.x
    p = optimal_p[i]
    q = optimal_q[i]
    pmf_array = np.array([ (1-q)*(1-p), p*(1-q), q*(1-p), p*q])
    optimal_pmf[i] = pmf_array
    optimal_values[i] = -result.fun  # Negate again to get the maximum value

# Output the results
"""
print(f'Optimal p: {optimal_p}')
print(f'Optimal q: {optimal_q}')
print(f'Maximum values of the objective function: {optimal_values}')
"""

print(f'Optimal PMF: {optimal_pmf}')


"""
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Load data
xs = np.load("res/xs.npy")
gains = np.load("res/gains.npy")
costs = np.load("res/costs.npy")

def optimize_p_q(gain_to_cost):
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
    expected_gains = p1 * gains[:, 0] + p2 * gains[:, 1] + p3 * gains[:, 2] + p4 * gains[:, 3]
    expected_costs = p1 * costs[:, 0] + p2 * costs[:, 1] + p3 * costs[:, 2] + p4 * costs[:, 3]
    #total_cost = np.sum(expected_costs)
    
    constraints.append( cp.sum(expected_costs) <= 100 ) #given a B value for budget constraint
    
    # Define the objective to maximize the net gain-to-cost ratio
    #objective = cp.Maximize(cp.sum(expected_gains * gain_to_cost - expected_costs)) #MAY BE MODIFIED!!
    objective = cp.Maximize(cp.sum(expected_gains))
    
    
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Return the optimal values of p1, p2, p3, and p4
    return p1.value, p2.value, p3.value, p4.value

#optimize p1, p2, p3, p4 values
p1_optimal, p2_optimal, p3_optimal, p4_optimal = optimize_p1234(gain_to_cost=6.375)
optimal_pmf = np.array([p1_optimal, p2_optimal, p3_optimal, p4_optimal]).T
print("Optimal PMF for pi:", optimal_pmf)

def get_policy_from_p1234(p1_optimal, p2_optimal, p3_optimal, p4_optimal):
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
"""
