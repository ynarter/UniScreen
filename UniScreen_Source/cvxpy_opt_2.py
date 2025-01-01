import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Load data
xs = np.load("res/xs.npy")
gains = np.load("res/gains.npy")
costs = np.load("res/costs.npy")

def optimize_p1234(gain_to_cost):
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