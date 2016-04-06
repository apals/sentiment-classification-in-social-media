import pylab as plt
import numpy as np

# Sample data
side = np.linspace(-2,2,15)
print side
print "-----------------"
X,Y = np.meshgrid(side,side)
print X
print "-----------------"
print Y
"-----------------"
Z = np.exp(-((X-1)**2+Y**2))
print Z

# Plot the density map using nearest-neighbor interpolation
plt.pcolormesh(X,Y,Z)
plt.show()