import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import beta
from collections import Counter
import os

independent_folder_path = "comparison/independent_data"
unified_folder_path = "comparison/unified_data"
xs_folder_path = "comparison/xs"

survival_disease_1 = np.load(os.path.join(independent_folder_path, "survival_disease_1.npy"))
cost_disease_1 = np.load(os.path.join(independent_folder_path, "cost_disease_1.npy"))
survival_disease_2 = np.load(os.path.join(independent_folder_path, "survival_disease_2.npy"))
cost_disease_2 = np.load(os.path.join(independent_folder_path, "cost_disease_2.npy"))

expected_gains = np.load(os.path.join(unified_folder_path, "expected_gains.npy"))
expected_costs = np.load(os.path.join(unified_folder_path, "expected_costs.npy"))

xs = np.load(os.path.join(xs_folder_path, "xs.npy"))
pmf_x = np.load(os.path.join(xs_folder_path, "pmf_x.npy"))

#For unified screening:
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



######For independent model:
def optimize_separately(x1, x2, pmf_x, expected_gains_x1, expected_costs_x1, expected_gains_x2, expected_costs_x2, k, B):
    N = x1.shape[0]
    a = 2 #number of possible combinations for policies
    
    #pmf = np1.ones(shape = pmf_x.shape)
    #pmf_x = pmf / 10000
    
    # Optimization variables (q's) for both x1 and x2
    q1 = cp.Variable(a * N)  #for disease 1 (x1)
    q2 = cp.Variable(a * N)  #for disease 2 (x2)

    gain_product_x1 = (expected_gains_x1 * pmf_x).T
    c1 = gain_product_x1.flatten()

    gain_product_x2 = (expected_gains_x2 * pmf_x).T
    c2 = gain_product_x2.flatten()

    constraints_1 = []
    constraints_2 = []
    constraints_1.append(q1 >= 0)  #non-negative q1
    constraints_2.append(q2 >= 0)  #non-negative q2

    cost_product_x1 = (expected_costs_x1 * pmf_x).T
    A1 = cost_product_x1.flatten()

    cost_product_x2 = (expected_costs_x2 * pmf_x).T
    A2 = cost_product_x2.flatten()

    constraints_1.append(A1 @ q1 <= B)  #cost constraint for disease 1
    constraints_2.append(A2 @ q2 <= B)  #cost constraint for disease 2

    O = np1.zeros((N, a * N))  #sum of q(a|x) = 1 constraint for both diseases
    for i in range(N):
        O[i, a * i : a * (i + 1)] = 1
    
    p = np1.ones(N)
    constraints_1.append(O @ q1 == p)  #constraint for disease 1
    constraints_2.append(O @ q2 == p)  #constraint for disease 2
    
    #solve the optimization problem
    prob1 = cp.Problem(cp.Maximize(c1.T @ q1), constraints_1)
    prob2 = cp.Problem(cp.Maximize(c2.T @ q2), constraints_2)

    prob1.solve()
    prob2.solve()

    q1_s = q1.value
    q2_s = q2.value
    
    #get the optimal policies for each disease
    optimal_qs_x1 = np1.zeros(N)
    optimal_qs_x2 = np1.zeros(N)

    for i in range(N):
        optimal_qs_x1[i] = np.argmax(q1_s[a * i : a * (i + 1)])
        optimal_qs_x2[i] = np.argmax(q2_s[a * i : a * (i + 1)])

    return optimal_qs_x1, optimal_qs_x2
   

#get the final policy based on the q's
optimal_qs_x1, optimal_qs_x2 = optimize_separately(
    xs[:, 0], xs[:, 1], pmf_x, survival_disease_1, cost_disease_1, survival_disease_2, cost_disease_2, 2, 5)

#get the final policy based on the q's FOR UNIFIED MODEL
policy = optimize_linear(xs, pmf_x, expected_gains, expected_costs, 2, 5)

""" print("Optimal qs for x1: ", optimal_qs_x1)
print("Optimal qs for x2: ", optimal_qs_x2) """

def get_independent_policy(optimal_qs_x1, optimal_qs_x2):
    optimal_qs_x1 = optimal_qs_x1.astype(int)
    optimal_qs_x2 = optimal_qs_x2.astype(int)
    
    #combine the independent decisions for x1 and x2 into a single policy
    independent_policy = (optimal_qs_x2 << 1) | optimal_qs_x1  #bitwise operation to encode policy
    return independent_policy

# Derive the independent policy
independent_policy = get_independent_policy(optimal_qs_x1, optimal_qs_x2)
              

def calculate_policy_differences(policy, independent_policy):
    policy_counts = Counter(policy)
    independent_policy_counts = Counter(independent_policy)

    for i in range(4):
        policy_counts[i] = policy_counts.get(i, 0)
        independent_policy_counts[i] = independent_policy_counts.get(i, 0)

    #calculate the differences for specified policies
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

folder_path = "comparison/policies_2"
os.makedirs(folder_path, exist_ok=True)

np.save(os.path.join(folder_path, "policy.npy"), policy)
np.save(os.path.join(folder_path, "independent_policy.npy"), independent_policy)
np.save(os.path.join(folder_path, "action_number_results.npy"), results)
np.save(os.path.join(folder_path, "optimal_qs_x1.npy"), optimal_qs_x1)
np.save(os.path.join(folder_path, "optimal_qs_x2.npy"), optimal_qs_x2)

print(f"Data saved successfully in '{folder_path}'!")

#plot the resulting policy assignments for our model
def plot_policy(policy):
    plt.figure()
    #plt.title("Unified Screening for Two Diseases")
    plt.scatter(xs[policy == 0, 0], xs[policy == 0, 1], color='silver', s=50, label='a = 1 (00)')
    plt.scatter(xs[policy == 1, 0], xs[policy == 1, 1], color='tab:pink', s=50, label='a = 2 (10)')
    plt.scatter(xs[policy == 2, 0], xs[policy == 2, 1], color='tab:purple', s=50, label='a = 3 (01)')
    plt.scatter(xs[policy == 3, 0], xs[policy == 3, 1], color='tab:red', s=50, label='a = 4 (11)')

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.ylabel(r"$x_2$", fontsize=18)
    plt.grid()
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

plot_policy(policy)

def plot_policy_separately(optimal_qs_x1, optimal_qs_x2, xs): 
    plt.figure(figsize=(6, 6))
    
    # Mask to exclude points on the edges
    mask = (xs[:, 0] > 0) & (xs[:, 0] < 1) & (xs[:, 1] > 0) & (xs[:, 1] < 1)
    
    # Plot for the four combinations of disease 1 (x1) and disease 2 (x2), excluding edges
    plt.scatter(xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 0) & mask, 0], 
                xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 0) & mask, 1], 
                color='silver', s=50, label='a = 1 (00)')
    plt.scatter(xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 0) & mask, 0], 
                xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 0) & mask, 1], 
                color='tab:pink', s=50, label='a = 2 (10)')
    plt.scatter(xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 1) & mask, 0], 
                xs[(optimal_qs_x1 == 0) & (optimal_qs_x2 == 1) & mask, 1], 
                color='tab:purple', s=50, label='a = 3 (01)')
    plt.scatter(xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 1) & mask, 0], 
                xs[(optimal_qs_x1 == 1) & (optimal_qs_x2 == 1) & mask, 1], 
                color='tab:red', s=50, label='a = 4 (11)')
    
    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_policy_separately(optimal_qs_x1, optimal_qs_x2, xs)