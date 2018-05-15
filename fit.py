"""
Here we are trying to learn how to fit a model to our data.

Simply running the code will generate some helpful plots and print out some information to the console.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress

# Define the model that we would like to use.
def model(x, a, b, c, d, e, f):
    return np.multiply(a*(1-np.exp(-(x-f)/b))*(np.exp(-(x-f)/c))*np.sin(2*np.pi*((x-f)-e)/d), [bool(i > f) for i in range(len(x))])

###############
## Dataset 1 ##
###############
# Load some pre-recorded data
data = json.load(open('./data/transducer_id_filter/transducer_id_filter.datafile', 'r'))
sys = data['sys1']
sys = sys[5:len(sys)//60]
x_data = np.array([i for i in range(len(sys))])
popt, pcov = curve_fit(model, x_data, sys, p0=(140, 105, 155, 14, 1, 100), maxfev=10000000)
print('Filtered Model fit: ', popt)

# Calculate the variance accounted for
y_pred = model(x_data, *popt)
y_bar = np.mean(sys)
SStot = np.sum((sys - y_bar)**2)
SSres = np.sum((sys - y_pred)**2)
r_squared = 1 - (SSres / SStot)
print('Variance accounted for: ', r_squared)

# Plot this data
plt.figure(1)
plt.plot(x_data/600000, sys, '.')
plt.plot(x_data/600000, model(x_data, *popt))
plt.xlabel('Lag (s)')
plt.ylabel('Response')
plt.title('Filtered Output\nVAF: %.4f' % r_squared)
plt.tight_layout()
# Optionally save the figure (takes a while, so only do this at the end.
# plt.savefig('./data/transducer_id_filter/model_fit.png', dpi=600, format='png')

# Dump the saved system into a file for access later.
#json.dump(sys, open('./data/transducer_id_filter/system.sys', 'w'))

###############
## Dataset 2 ##
###############
# Load some pre-recorded data
data = json.load(open('./data/transducer_id/transducer_id.datafile', 'r'))
sys = data['sys1']
sys = sys[5:len(sys)//60]
x_data = np.array([i for i in range(len(sys))])
popt, pcov = curve_fit(model, x_data, sys, p0=(140, 105, 155, 14, 1, 113), maxfev=10000000)
print('Non-filtered model fit: ', popt)

# Calculate the variance accounted for
y_pred = model(x_data, *popt)
y_bar = np.mean(sys)
SStot = np.sum((sys - y_bar)**2)
SSres = np.sum((sys - y_pred)**2)
r_squared = 1 - (SSres / SStot)
print('Variance accounted for: ', r_squared)

# Plot this data
plt.figure(2)
plt.plot(x_data/600000, sys)
plt.xlabel('Lag (s)')
plt.ylabel('Response')
plt.title('System Response' % r_squared)
plt.tight_layout()
# Optionally save the figure (takes a while, so only do this at the end.
plt.savefig('./data/transducer_id/system.png', dpi=600, format='png')


# Plot this data
plt.figure(3)
plt.plot(x_data/600000, sys, '.')
plt.plot(x_data/600000, model(x_data, *popt))
plt.xlabel('Lag (s)')
plt.ylabel('Response')
plt.title('System Response\nVAF: %.4f' % r_squared)
plt.tight_layout()
# Optionally save the figure (takes a while, so only do this at the end.
plt.savefig('./data/transducer_id/model_fit.png', dpi=600, format='png')

# Dump the saved system into a file for access later.
json.dump(sys, open('./data/transducer_id/system.sys', 'w'))

# Show the plots
#plt.show()
