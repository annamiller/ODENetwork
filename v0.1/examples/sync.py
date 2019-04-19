"""
used to test stdp sychronization.

one to one connection
different types of currents such as poisson spikes, constant currents,
periodic spikes and sine waves, were injected to both pre- and post-synaptic
neuron with different spiking frequencies.
"""
import numpy as np
import sys
sys.path.append('..')
import networks # noqa: E402
import neuron_models as nm # noqa: E402
import experiments as ex # noqa: E402
import lab_manager as lm # noqa: E402

# Step 1: Pick a network and visualize it
neuron_nums = [1, 1]  # number of neurons in each layer
net = networks.get_multilayer_fc(
    nm.Soma, nm.SynapseWithDendrite, neuron_nums)
networks.draw_layered_digraph(net)

# step 2: design an experiment. four different shapes of input

# # poisson
# i_max = 55.
# num_sniffs = 20
# time_per_sniff = 200.
# total_time = num_sniffs*time_per_sniff
# base_rate = 0.025
# ex.feed_gaussian_rate_poisson_spikes_DL(
#     net, base_rate, i_max=i_max, num_sniffs=num_sniffs,
#     time_per_sniff=time_per_sniff)

# constant
num_layers = 2
neuron_inds = [[0], [0]]
# adjust post-synaptic DC current to set initial post-synaptic frequency
current_vals = [[2.5], [14]]
total_time = 1000
ex.const_current(net, num_layers, neuron_inds, current_vals)

# # periodic spikes
# total_time = 1000
# i_max = 100.
# base_rate = 0.01
# ratio = 0.95
# ex.feed_periodic_spikes(net, base_rate, ratio, total_time, i_max)

# # sine wave
# i_max = 25
# num_layers = 2
# neuron_inds = [[0], [0]]
# rates = [[0.5], [0.55]]
# total_time = 200
# ex.sine_wave(net, num_layers, neuron_inds, rates, i_max)

# step 3: ask our lab manager to set up the lab for the experiment.
f, initial_conditions, _ = lm.set_up_lab(net)

# step 4: run the lab and gather data
time_sampled_range = np.arange(0., total_time, 0.1)
data = lm.run_lab(f, initial_conditions, time_sampled_range)

# step 5: plot
# for layer_idx in range(len(net.layers)):
#     lm.show_all_neuron_in_layer(
#         time_sampled_range, data, net, layer_idx)
lm.show_one_to_one_neuron_together(time_sampled_range, data, net)

for layer_idx in range(len(net.layers)):
    lm.show_all_dendrite_onto_layer(
        time_sampled_range, data, net, layer_idx, delta_time=None)
