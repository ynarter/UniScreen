import jax
import jax.numpy as np2
import numpy as np

#10x10 joint pmf for x1 and x2
x1 = np.linspace(0, 1, 10)
x2 = np.linspace(0, 1, 10)

#joint pmf, ASSUME uniform distribution for now
joint_pmf = np.ones((10, 10)) / 100

joint_pmf /= joint_pmf.sum()
print("Joint pmf for x1 and x2:", joint_pmf)

def sample_from_pmf(pmf, key):
    pmf_flat = pmf.flatten()
    idx = jax.random.choice(key, np.arange(pmf.size), p=pmf_flat, shape=(10,)) #make a random choice, given the sample number N
    print(idx)
    x1_idx = idx // pmf.shape[1] #determine the row/column
    x2_idx = idx % pmf.shape[1]
    x1_vals = x1[x1_idx]
    x2_vals = x2[x2_idx]
    xs = np2.stack([x1_vals, x2_vals], axis=-1)
    return xs

# Updated part of the original simulation code
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
xs = sample_from_pmf(joint_pmf, subkey)
print(xs)