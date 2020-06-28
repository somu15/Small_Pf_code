import os
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import Utils
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


rnd = np.random.normal(loc=0,scale=1.0,size=100)
kde = Utils.Kernel_1D(kernel_name='gaussian')


x = np.arange(-3,3,0.01)
req = np.zeros(len(x))

for ind in np.arange(0,len(x),1):
    req[ind] = kde.density(rvs = rnd, value = x[ind],bandwidth_rule='silverman')
plt.plot(x,req)

# kde_ref = KernelDensity(kernel='gaussian',bandwidth=np.std(rnd)).fit(rnd.reshape(-1, 1))
# log_dens = kde_ref.score_samples(x.reshape(-1, 1))
#
# plt.plot(x,np.exp(log_dens))

plt.show()
