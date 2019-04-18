"""
used to test stdp learning.

multiple pre-synaptic neurons
different possion spike trains were injected to pre-synaptic neurons only.
"""
import numpy as np
import sys
sys.path.append('..')
import networks # noqa: E402
import neuron_models as nm # noqa: E402
import experiments as ex # noqa: E402
import lab_manager as lm # noqa: E402

# Step 1: Pick a network and visualize it
neuron_nums = [2, 1]  # number of neurons in each layer
# use SomaWithAmpa instead to strengthen somatic spikes
net = networks.get_multilayer_fc(
    nm.SomaWithAmpa, nm.SynapseWithDendrite, neuron_nums)
networks.draw_layered_digraph(net)

# step 2: design an experiment. (Fixing input currents really)

# poisson
i_max = 55.
num_sniffs = 20
time_per_sniff = 200.
total_time = num_sniffs*time_per_sniff
base_rate = 0.025
ex.feed_gaussian_rate_poisson_spikes(
    net, base_rate, i_max=i_max, num_sniffs=num_sniffs,
    time_per_sniff=time_per_sniff)
# # for more than two pre-synaptic neurons, use the following instead
# ex.feed_gaussian_rate_poisson_spikes_DL(
#     net, base_rate, i_max=i_max, num_sniffs=num_sniffs,
#     time_per_sniff=time_per_sniff)

# periodic
# total_time = 1000
# i_max=100.
# base_rate = 0.01
# ratio = 0.95
# ex.feed_periodic_spikes(net, base_rate, ratio, total_time, i_max)


# step 3: ask our lab manager to set up the lab for the experiment.
f, initial_conditions, _ = lm.set_up_lab(net)

# step 4: run the lab and gather data
time_sampled_range = np.arange(0., total_time, 0.1)
data = lm.run_lab(f, initial_conditions, time_sampled_range)

# Time to witness some magic
for layer_idx in range(len(net.layers)):
    lm.show_all_neuron_in_layer(
        time_sampled_range, data, net, layer_idx)

for layer_idx in range(len(net.layers)):
    lm.show_all_dendrite_onto_layer(
        time_sampled_range, data, net, layer_idx, delta_time=None)
