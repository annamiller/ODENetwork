"""
This script uses a package called Brian. It is developed with the intention
of modeling biologically realisting (or not) neural networks. For more info
and install instructions, see: https://brian2.readthedocs.io/en/stable/


Before running, please make sure there is a folder created matching the
prefix variable in your working directory.
"""

import numpy as np
import matplotlib.pyplot as plt

from brian2 import *

from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import itertools
from itertools import cycle

import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from sklearn.metrics import mutual_info_score

plt.style.use('ggplot')

nu = -1.5*mV
a = 0.7
b = 0.8
t1 = 0.08*ms
t2 = 3.1*ms

# Where to save data
prefix = 'WLC_data/'

# start scopes makes it such that any previous runs info are not used
start_scope()

# equations to describe neuron behavior - Fitzo Nagumo
eqns = '''
I_inj : amp
I_syn : 1
dV/dt = (V-(V**3)/(3*mV**2) - w - z*(V - nu) + 0.35*mV + I_inj*Mohm)/t1 : volt
dw/dt = (V - b*w + a*mV)/ms : volt
dz/dt = (I_syn - z)/t2 : 1
'''

g_syn = 0.1 #needs to be scaled with size of network

# synaptic current
syn = '''
I_syn_post = g_syn/(1.0+exp(-1000*V_pre/mV)): 1 (summed)
'''

defaultclock.dt = .01*ms
N = 1000

# Function to define which neurons get current and how much
def get_rand_I(p = 0.33, I_inp = 0.15):
    I = np.random.random(N)
    I[I > p] = 0
    I[np.nonzero(I)] = I_inp
    return I

# Creates a neuron group with N neurons, the 'threshold' for spiking
G = NeuronGroup(N, eqns, threshold='V > 1*mV', method = 'rk4')

# Define a state monitor that stores the values of the specified variables for all neurons
trace = StateMonitor(G, ['V', 'w', 'z'], record=True)

# Define a spike monitor that records when spikes happen in the network
spikes = SpikeMonitor(G)

# Defines the types of connections (what two groups and what model to be used)
S = Synapses(G, G, model = syn)
# Actually makes he connections with probability 0.5
S.connect(p = 0.5)

# Set initial contitions for the state variables of all neurons
G.V = -1.2*mV
G.w = -0.62*mV
G.z = 0.0

# Stores this state of the network (so it can be used in multiple ways)
store()

c = ['k', 'r', 'b']
fig = plt.figure(figsize = (10,7))
for i in range(3):
    # restores the stored state of the network
    restore()
    I = get_rand_I(p = 0.33, I_inp = 0.15)

    G.I_inj = I*nA

    # runs the network for the specified time
    run(100*ms)
    np.save(prefix+'spikes_t_'+str(i) ,spikes.t/ms)
    np.save(prefix+'spikes_i_'+str(i) ,spikes.i)
    np.save(prefix+'I_'+str(i), I)
    np.save(prefix+'trace_V_'+str(i), trace.V)
    np.save(prefix+'trace_t_'+str(i), trace.t)
    plt.plot(spikes.t/ms, spikes.i, ',', color = c[i])
plt.title('3 Odors', fontsize = 22)
plt.ylabel('Neuron Num', fontsize = 16)
plt.xlabel('Time', fontsize = 16)
fig.savefig(prefix+str(N)+'_neuron.pdf', bbox_inches = 'tight')



# Everything else below is used to make plots
def load_wlc_data(num_runs = 3):
    spikes_t_arr = []
    spikes_i_arr = []
    I_arr = []
    trace_V_arr = []
    trace_t_arr = []
    for i in range(num_runs):
        spikes_t_arr.append(np.load(prefix+'spikes_t_'+str(i)+'.npy'))
        spikes_i_arr.append(np.load(prefix+'spikes_i_'+str(i)+'.npy'))
        I_arr.append(np.load(prefix+'I_'+str(i)+'.npy'))
        trace_V_arr.append(np.load(prefix+'trace_V_'+str(i)+'.npy'))
        trace_t_arr.append(np.load(prefix+'trace_t_'+str(i)+'.npy'))
    return spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr

spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr = load_wlc_data(num_runs = 3)
N = len(I_arr[0])

# Local Field potential
fig = plt.figure(figsize = (10,7))
plt.plot(trace_t_arr[0]/ms, np.mean(trace_V_arr[0]/mV, axis = 0))
plt.title('LFP for WLC ' + str(N) + ' Neurons', fontsize = 20)
plt.xlabel('Time (ms)', fontsize = 16)
plt.ylabel('LFP (mV)', fontsize = 16)
fig.savefig('LFP_WLC_' + str(N) + '.pdf', bbox_inches = 'tight')

#PCA
k = 3 #three principle components

length = len(trace_V_arr[0][0])

data = np.hstack(trace_V_arr)

# svd decomposition and extract eigen-values/vectors
pca = PCA(n_components=k)
pca.fit(data.T)
wk = pca.explained_variance_
vk = pca.components_

# Save the pca data into each odor/conc
Xk = pca.transform(data.T)

pca1 = Xk[0:length]
pca2 = Xk[length:2*length]
pca3 = Xk[2*length:3*length]

#Plot PCA
start = 400

fig = plt.figure(figsize = (10,7)).gca(projection='3d')

skip = 3
name = [pca1[::skip], pca2[::skip], pca3[::skip]]
for j in range(3):
    fig.scatter(name[j][start:,0], name[j][start:,1], name[j][start:,2],s=10)

plt.title('PCA ' + str(N) + ' neuron')
# ax.view_init(-30, 45)
fig.view_init(elev = 30)
fig.figure.savefig('PCA_' + str(N) + '.pdf', bbox_inches = 'tight')
fig.figure.savefig('PCA_' + str(N) + '.png', bbox_inches = 'tight', dpi = 400)
