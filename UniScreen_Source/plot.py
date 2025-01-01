import matplotlib.pyplot as plt
import numpy as np

xs = np.load("res/xs.npy")
gains = np.load("res/gains.npy")
costs = np.load("res/costs.npy")
print("Gains:", gains)
print(gains.shape)
print("Costs:", costs)
print(costs.shape)


def get_policy(gain_to_cost):
    policy = np.argmax(gains * gain_to_cost - costs, axis=-1)
    budget = np.array([costs[i, policy[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
    print("Policy:", policy )
    return policy, budget

policy, budget = get_policy(gain_to_cost=6.375)
print(budget)

# 00 10 01 11
def get_policy_conditionalx0(gain_to_cost):
    _gains = gains[:,[0,1]]
    _costs = costs[:,[0,1]]
    policy = np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    print("Policy x0:", policy )
    return policy

def get_policy_conditionalx1(gain_to_cost):
    _gains = gains[:,[2,3]]
    _costs = costs[:,[2,3]]
    policy = 2 + np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    print("Policy x1:", policy )
    return policy

def get_policy_conditional0x(gain_to_cost):
    _gains = gains[:,[0,2]]
    _costs = costs[:,[0,2]]
    policy = 2 * np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    print("Policy 0x:", policy )
    return policy

def get_policy_conditional1x(gain_to_cost):
    _gains = gains[:,[1,3]]
    _costs = costs[:,[1,3]]
    policy = 1 + 2 * np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    print("Policy 1x:", policy )
    return policy

policy_conditionalx0 = get_policy_conditionalx0(gain_to_cost=10)
policy_conditionalx1 = get_policy_conditionalx1(gain_to_cost=10)
policy_conditional0x = get_policy_conditional0x(gain_to_cost=10)
policy_conditional1x = get_policy_conditional1x(gain_to_cost=10)

policy_conditionalx0 = (policy_conditionalx0 == 1)
policy_conditionalx1 = (policy_conditionalx1 == 3)
policy_conditional0x = (policy_conditional0x == 2)
policy_conditional1x = (policy_conditional1x == 3)

yea_0 = policy_conditionalx0 & policy_conditionalx1
nay_0 = ~policy_conditionalx0 & ~policy_conditionalx1
yea_1 = policy_conditional0x & policy_conditional1x
nay_1 = ~policy_conditional0x & ~policy_conditional1x

_00 = nay_0 & nay_1
_10 = yea_0 & nay_1
_01 = nay_0 & yea_1
_11 = yea_0 & yea_1

_11 = _11 | (yea_0 & policy_conditional1x)
_11 = _11 | (yea_1 & policy_conditionalx1)

_00 = _00 | (nay_0 & ~policy_conditional0x)
_00 = _00 | (nay_1 & ~policy_conditionalx0)

_10 = _10 | (yea_0 & ~policy_conditional1x)
_10 = _10 | (nay_1 & policy_conditionalx0)

_01 = _01 | (nay_0 & policy_conditional0x)
_01 = _01 | (yea_1 & ~policy_conditionalx0)

plt.scatter(xs[_00, 0], xs[_00, 1], color='silver', s=50)
plt.scatter(xs[_10, 0], xs[_10, 1], color='tab:pink', s=50)
plt.scatter(xs[_01, 0], xs[_01, 1], color='tab:purple', s=50)
plt.scatter(xs[_11, 0], xs[_11, 1], color='tab:red', s=50)

plt.axis('scaled')
plt.title("Plot 1")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.show()
plt.tight_layout()
plt.savefig('figs/points.pdf')

def get_policy_conditionalxx(gain_to_cost):
    _gains = gains[:,[0,3]]
    _costs = costs[:,[0,3]]
    policy = 3 * np.argmax(_gains * gain_to_cost - _costs, axis=-1)
    return policy

policy_conditionalxx = get_policy_conditionalxx(gain_to_cost=10)

def get_policy_independent(budget):

    for threshold0 in np.linspace(0, 1, 1000):
        _budget = costs[xs[:,0] > threshold0, 1].sum() / xs.shape[0]
        if _budget < budget / 2:
            break

    for threshold1 in np.linspace(0, 1, 1000):
        _budget = costs[xs[:,1] > threshold1, 2].sum() / xs.shape[0]
        if _budget < budget / 2:
            break

    policy = np.zeros(xs.shape[0])
    policy += np.where(xs[:,0] > threshold0, 1, 0)
    policy += np.where(xs[:,1] > threshold1, 2, 0)
    
    return policy, np.array([threshold0, threshold1])

policy_independent, thresholds = get_policy_independent(budget)

def plot(policy):

    plt.scatter(xs[policy == 0, 0], xs[policy == 0, 1], color='silver', s=50)
    plt.scatter(xs[policy == 1, 0], xs[policy == 1, 1], color='tab:pink', s=50)
    plt.scatter(xs[policy == 2, 0], xs[policy == 2, 1], color='tab:purple', s=50)
    plt.scatter(xs[policy == 3, 0], xs[policy == 3, 1], color='tab:red', s=50)

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()
    plt.show()

    plt.tight_layout()
    plt.savefig('figs/points.pdf')

def plot_diff(policy, _policy):

    plt.scatter(xs[(policy == 0) & (_policy != 0), 0], xs[(policy == 0) & (_policy != 0), 1], color='silver', s=50)
    plt.scatter(xs[(policy == 1) & (_policy != 1), 0], xs[(policy == 1) & (_policy != 1), 1], color='tab:pink', s=50)
    plt.scatter(xs[(policy == 2) & (_policy != 2), 0], xs[(policy == 2) & (_policy != 2), 1], color='tab:purple', s=50)
    plt.scatter(xs[(policy == 3) & (_policy != 3), 0], xs[(policy == 3) & (_policy != 3), 1], color='tab:red', s=50)

    plt.axis('scaled')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid()

    plt.tight_layout()
    plt.savefig('figs/points.pdf')

# plot_diff(policy_independent, policy)

plot(policy)

# benefit = np.array([gains[i, policy[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
# spending = np.array([costs[i, policy[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
# print(benefit / spending)

# policy_independent = policy_independent.astype(int)

# benefit = np.array([gains[i, policy_independent[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
# spending = np.array([costs[i, policy_independent[i]] for i in range(xs.shape[0])]).sum() / xs.shape[0]
# print(benefit / spending)
