import socket
import spynnaker8 as p
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility import Timer
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import pdb
import numpy as np

timer = Timer()

dt = 1          # (ms) simulation timestep
tstop = 2000      # (ms) simulaton duration
delay = 2

n_l1 = 1 # number of cells in layer 1
n_l2 = 1 # number of cells in layer 2
n_total = n_l1+n_l2

# === Build the network ===

node_id = p.setup(timestep=dt, min_delay=delay, max_delay=delay)

#  100 neurons per core
p.set_number_of_neurons_per_core(p.IF_curr_exp, 100)
node_p = 1

cell_params = {'tau_m': 20.0,
               'tau_syn_E': 5.0,
               'tau_syn_I': 5.0,
               'v_rest': -65.0,
               'v_reset': -65.0,
               'v_thresh': -50.0,
               'tau_refrac': 0.1,
               'cm': 1,
               'i_offset': 0.0
               }

print(cell_params)

timer.start()

# Populations
print("%s Creating cell populations..." % node_id)
celltype = p.IF_curr_exp
cells_l1 = p.Population(n_l1, celltype(**cell_params), label="Layer_1")
cells_l2 = p.Population(n_l2, celltype(**cell_params), label="Layer_2")

f = 200
spike_train_1=p.SpikeSourceArray(spike_times=[f*1,f*2,f*4,f*8,f*16,f*32,f*64])
i_spikes_1=p.Population(1,spike_train_1)

# Connectivity
print("%s Connecting populations..." % node_id)
cell_conn = p.AllToAllConnector()
w = 0.0525
connections = { 'i1l1': p.Projection(i_spikes_1, cells_l1, cell_conn,
                        receptor_type='excitatory',
                        synapse_type=p.StaticSynapse(weight=w, delay=delay))}

# === Setup recording ===
print("%s Setting up recording..." % node_id)
cells_l1.record(["v","spikes"])
cells_l2.record(["v","spikes"])
i_spikes_1.record(["spikes"])

buildCPUTime = timer.diff()

# === Run simulation ===
print("%d Running simulation..." % node_id)



p.run(tstop)

simCPUTime = timer.diff()

# === Print results to file ===

l1_voltage = cells_l1.get_data("v")
l1_spikes = cells_l1.get_data("spikes")
l2_spikes = cells_l2.get_data("spikes")
in_spikes = i_spikes_1.get_data("spikes")

v_array = np.array(l1_voltage.segments[0].filter(name="v")[0]).reshape(-1)
print("\n\n\n*********** v max = %0.3f *********** \n\n\n" %(max(v_array)))
np.savetxt("v_spinnaker.csv", v_array, delimiter=",")

plt.plot(v_array)
plt.savefig("Voltage.png")
# plt.show()

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(in_spikes.segments[0].spiketrains, xlabel="Time/ms", xticks=True,
          yticks=True, markersize=1, xlim=(0, tstop)),
    title="Input Spikes",
    annotations="Simulated with {}".format(p.name())
)
plt.savefig("InputSpikes.png")
# plt.show()

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(l1_spikes.segments[0].spiketrains, xlabel="Time/ms", xticks=True,
          yticks=True, markersize=1, xlim=(0, tstop)),
    title="Output Spikes L1",
    annotations="Simulated with {}".format(p.name())
)
plt.savefig("OutputSpikesL1.png")
# plt.show()

writeCPUTime = timer.diff()

if node_id == 0:
    print("\n--- Simulation Summary ---")
    print("Nodes                  : %d" % node_p)
    print("Number of Neurons      : %d" % n_total)
    print("Number of Synapses     : %s" % connections)
    print("Build time             : %g s" % buildCPUTime)
    print("Simulation time        : %g s" % simCPUTime)
    print("Writing time           : %g s" % writeCPUTime)

# === Finished with simulator ===

p.end()
