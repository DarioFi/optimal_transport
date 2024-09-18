from angles_bound import N_sphere_cap, theta_bound
import matplotlib.pyplot as plt
import numpy as np

# Define the range of alpha values
N = 4

alpha_vals = np.linspace(0, .999, 1000)

# Calculate the corresponding N values

angles = [theta_bound(alpha) / 2 for alpha in alpha_vals]

N_vals = [N_sphere_cap(angle, N) for angle in angles]

print(N_vals[500])

# Plot the results
plt.plot(alpha_vals, N_vals)
plt.yscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$N(\alpha)$')
plt.title(fr'$N(\alpha)$ for $N = {N}$')
plt.show()
