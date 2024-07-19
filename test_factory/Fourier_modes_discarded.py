import numpy as np
from scipy.stats import loguniform


N = int(1e6)
hi = 2.0 * np.pi / 4.0
low = 2.0 * np.pi / 512
psi = 65 / 180.0 * np.pi
k_x = loguniform.rvs(low, hi, size=N)
k_y = loguniform.rvs(low, hi, size=N)
k_z = loguniform.rvs(low, hi, size=N)

# k_x = np.random.uniform(low=low, high=hi, size=N)
# k_y = np.random.uniform(low=low, high=hi, size=N)
# k_z = np.random.uniform(low=low, high=hi, size=N)

counter = 0
for pair in zip(k_x, k_y, k_z):
    if pair[0] > 0.05 and pair[0] > np.sqrt(pair[1] ** 2 + pair[2] ** 2) * np.tan(psi):
        counter += 1

print(1 - counter / N)
