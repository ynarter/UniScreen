import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Parameters for the Beta distributions
alpha1, beta1 = 1, 1  # Parameters for X1
alpha2, beta2 = 1, 1  # Parameters for X2

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

# Print or visualize the joint PMF
print("Joint PMF shape:", joint_pmf.shape)

plt.imshow(joint_pmf, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
plt.colorbar(label='Probability')
plt.xlabel('$X_1$')
plt.ylabel('$X_2$')
plt.title('Joint PMF of $X_1$ and $X_2$')
plt.show()