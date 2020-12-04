
from numpy import genfromtxt
import matplotlib.pyplot as plt

v_spinnaker = genfromtxt('v_spinnaker.csv', delimiter=',')
v_norse = genfromtxt('v_norse.csv', delimiter=',')

# Plotting output spikes
plt.plot(v_spinnaker, label="spinnaker")
plt.plot(v_norse, label="norse")
plt.legend(loc="upper right")
plt.show()
