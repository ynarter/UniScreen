import jax
import jax.numpy as np
import numpy as np1
from jax.scipy.special import logsumexp
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.stats import beta
from collections import Counter
import os
from matplotlib.lines import Line2D
import itertools

unified_folder_path = "comparison/unified_data"
xs_folder_path = "comparison/xs"

expected_gains = np.load(os.path.join(unified_folder_path, "expected_gains.npy"))
expected_costs = np.load(os.path.join(unified_folder_path, "expected_costs.npy"))

xs = np.load(os.path.join(xs_folder_path, "xs.npy"))
pmf_x = np.load(os.path.join(xs_folder_path, "pmf_x.npy"))

def plot_expected_gains_and_costs(xs, expected_gains, expected_costs):
    #reshape the xs into individual x1 and x2 for plotting
    x1_vals = xs[:, 0]
    x2_vals = xs[:, 1]
    
    policies = ['00', '10', '01', '11']
    
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    for i in range(4):
        sc1 = axs[0, i].scatter(x1_vals, x2_vals, c=expected_gains[:, i], cmap='coolwarm', s=10)
        axs[0, i].set_title(f"Expected Survival Time - Policy {policies[i]} (a= {i+1})")
        axs[0, i].set_xlabel(r"$x_1$")
        axs[0, i].set_ylabel(r"$x_2$")
        plt.colorbar(sc1, ax=axs[0, i])
        
        sc2 = axs[1, i].scatter(x1_vals, x2_vals, c=expected_costs[:, i], cmap='viridis', s=10)
        axs[1, i].set_title(f"Expected Cost - Policy {policies[i]} (a= {i+1})")
        axs[1, i].set_xlabel(r"$x_1$")
        axs[1, i].set_ylabel(r"$x_2$")
        plt.colorbar(sc2, ax=axs[1, i])

    plt.tight_layout()
    plt.show()
    


def plot_gains_minus_costs(xs, expected_gains, expected_costs, lambda_val):
    x1_vals = xs[:, 0]
    x2_vals = xs[:, 1]
    
    gain_minus_cost = expected_gains - lambda_val * expected_costs
    policies = ['00', '10', '01', '11']
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    for i in range(4):
        sc = axs[i].scatter(x1_vals, x2_vals, c=gain_minus_cost[:, i], cmap='coolwarm', s=10)
        axs[i].set_title(f"Gain - λ * Cost: Policy {policies[i]} (a={i+1})")
        axs[i].set_xlabel(r"$x_1$")
        axs[i].set_ylabel(r"$x_2$")
        plt.colorbar(sc, ax=axs[i])

    plt.tight_layout()
    plt.show()

def plot_gains_costs_with_decision_boundary(xs, expected_gains, expected_costs, lambda_val):
    x1_vals = xs[:, 0]
    x2_vals = xs[:, 1]
    
    gain_minus_cost = expected_gains - lambda_val * expected_costs
    max_gain_minus_cost = np.max(gain_minus_cost, axis=1)
    best_policy = np.argmax(gain_minus_cost, axis=1)
    best_policy += 1  # Add 1 to match the policy labels 1, 2, 3, 4
    
    policies = ['00', '10', '01', '11']
    for i in range(4):
        #fig = plt.figure(figsize=(8, 6))
        sc = plt.scatter(x1_vals, x2_vals, c=gain_minus_cost[:, i], cmap='magma', s=10)
        #plt.title(f"Survival Time - λ * Cost: Policy {policies[i]}")
        plt.axis('scaled')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")
        plt.colorbar(sc)
        plt.tight_layout()
        plt.show()

    #fig_max = plt.figure(figsize=(8, 6))
    sc_max = plt.scatter(x1_vals, x2_vals, c=max_gain_minus_cost, cmap='plasma', s=10)
    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$", fontsize=12)
    plt.ylabel(r"$x_2$", fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=14)
    cl = plt.colorbar(sc_max)
    cl.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()

    fig_boundary = plt.figure(figsize=(8, 6))
    sc_boundary = plt.scatter(x1_vals, x2_vals, c=best_policy, cmap='tab10', s=50)
    #plt.title("Decision Boundary (Best Policy)")
    bounds = [0, 1, 2, 3, 4]
    #cl=plt.colorbar(sc_boundary, ticks=[1, 2, 3, 4])  # Show ticks 1, 2, 3, 4 on colorbar
    #cl.ax.tick_params(labelsize=12)
    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r"$x_1$", fontsize=16)
    plt.ylabel(r"$x_2$", fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    colors = plt.cm.tab10(np.linspace(0, 1, 4))
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], label='a=1 (00)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], label='a=2 (10)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[2], label='a=3 (01)', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[3], label='a=4 (11)', markersize=10)
    ]
    plt.legend(handles=legend_elements, fontsize=16, loc="upper left")
    #plt.grid()
    #plt.legend(fontsize=14, loc="upper left")
    plt.tight_layout()
    plt.show()

#to plot the survival time for a fixed x2 to observe behavior wrt x1
def plot_x1_vs_gain_for_policies(xs, expected_gains):
    x1_values = xs[:, 0]  # Extract x1 values
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))  # Create a 2x2 grid of subplots
    policies = ['Policy 00', 'Policy 10', 'Policy 01', 'Policy 11']
    
    for i in range(4):
        #gain_minus_cost = expected_gains[:, i] - lambda_val * expected_costs[:, i]  # Gain - Cost for each policy
        gain = expected_gains[:, i]
        row = i // 2  # Determine the row in the 2x2 grid
        col = i % 2   # Determine the column in the 2x2 grid
        
        ax[row, col].scatter(x1_values, gain, c='blue', alpha=0.5)
        ax[row, col].set_xlabel(r"$x_1$")
        ax[row, col].set_ylabel('Expected Survival Time')
        ax[row, col].set_title(policies[i])
        ax[row, col].grid(True)
    
    plt.tight_layout()
    plt.show()
        

plot_expected_gains_and_costs(xs, expected_gains, expected_costs) #plot surv times and costs for each policy

lambda_val = 0.25  #choose a lambda
plot_gains_minus_costs(xs, expected_gains, expected_costs, lambda_val)

plot_gains_costs_with_decision_boundary(xs, expected_gains, expected_costs, lambda_val)

plot_x1_vs_gain_for_policies(xs, expected_gains)

################################################################################################################

def plot_policy_differences(xs, expected_gains, expected_costs):
    x1_vals = xs[:, 0]
    x2_vals = xs[:, 1]

    policies = ['00', '10', '01', '11']
    policy_pairs = list(itertools.combinations(range(4), 2))  # All unique pairs (i, j)

    fig, axs = plt.subplots(2, len(policy_pairs), figsize=(5 * len(policy_pairs), 10))

    for idx, (i, j) in enumerate(policy_pairs):
        diff_gain = expected_gains[:, j] - expected_gains[:, i]
        sc1 = axs[0, idx].scatter(x1_vals, x2_vals, c=diff_gain, cmap='coolwarm', s=10)
        axs[0, idx].set_title(f"Δ Survival Time: {policies[j]} - {policies[i]}")
        axs[0, idx].set_xlabel(r"$x_1$")
        axs[0, idx].set_ylabel(r"$x_2$")
        plt.colorbar(sc1, ax=axs[0, idx])

        diff_cost = expected_costs[:, j] - expected_costs[:, i]
        sc2 = axs[1, idx].scatter(x1_vals, x2_vals, c=diff_cost, cmap='PiYG', s=10)
        axs[1, idx].set_title(f"Δ Cost: {policies[j]} - {policies[i]}")
        axs[1, idx].set_xlabel(r"$x_1$")
        axs[1, idx].set_ylabel(r"$x_2$")
        plt.colorbar(sc2, ax=axs[1, idx])

    plt.tight_layout()
    plt.show()
    


def plot_selected_differences(xs, expected_gains, expected_costs, lambd=1.0):
    x1_vals = xs[:, 0]
    x2_vals = xs[:, 1]

    selected_pairs = [
        (1, 0),  # 2 - 1
        (3, 1),  # 4 - 2
        (3, 0),  # 4 - 1
    ]
    policies = ['00', '10', '01', '11']

    fig, axs = plt.subplots(2, len(selected_pairs), figsize=(5 * len(selected_pairs), 10))

    for idx, (i, j) in enumerate(selected_pairs):
        # Difference in survival time
        diff_gain = expected_gains[:, i] - expected_gains[:, j]
        sc1 = axs[0, idx].scatter(x1_vals, x2_vals, c=diff_gain, cmap='coolwarm', s=10)
        axs[0, idx].set_title(f"Δ Survival: {policies[i]} - {policies[j]}")
        axs[0, idx].set_xlabel(r"$x_1$")
        axs[0, idx].set_ylabel(r"$x_2$")
        plt.colorbar(sc1, ax=axs[0, idx])

        # Difference in cost
        diff_cost = expected_costs[:, i] - expected_costs[:, j]
        sc2 = axs[1, idx].scatter(x1_vals, x2_vals, c=diff_cost, cmap='PiYG', s=10)
        axs[1, idx].set_title(f"Δ Cost: {policies[i]} - {policies[j]}")
        axs[1, idx].set_xlabel(r"$x_1$")
        axs[1, idx].set_ylabel(r"$x_2$")
        plt.colorbar(sc2, ax=axs[1, idx])

    plt.tight_layout()
    plt.show()

    fig2, axs2 = plt.subplots(1, len(selected_pairs), figsize=(5 * len(selected_pairs), 5))
    for idx, (i, j) in enumerate(selected_pairs):
        kappa_i = expected_gains[:, i] - lambd * expected_costs[:, i]
        kappa_j = expected_gains[:, j] - lambd * expected_costs[:, j]
        diff_kappa = kappa_i - kappa_j

        sc = axs2[idx].scatter(x1_vals, x2_vals, c=diff_kappa, cmap='bwr', s=10)
        axs2[idx].set_title(f"Δ Kappa: {policies[i]} - {policies[j]}")
        axs2[idx].set_xlabel(r"$x_1$")
        axs2[idx].set_ylabel(r"$x_2$")
        plt.colorbar(sc, ax=axs2[idx])

    plt.tight_layout()
    plt.show()

plot_policy_differences(xs, expected_gains, expected_costs) 
plot_selected_differences(xs, expected_gains, expected_costs, lambd=0.5)
