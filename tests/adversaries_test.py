import sys
import numpy as np 

sys.path.insert(0, "../modules")
from topology_generation import Topology
from adversaries import Adversaries


np.random.seed(0)

# Parameters Set-up for Topology class
topo_type = 'r-robust'
n_nodes = 100
period = 1
hyperparams_dict = {'threshold': 0.20, 'robustness': 2}
# Get adjacency matrix
topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
adjacency = topology.adjacencies[0]


# Parameters Set-up for Adversaries class
F = 3
n_dims = 2
display_flag = False

# states construction for test
states = np.random.randn(n_nodes, n_dims)

### Test Adversaries: F-total Model
adv_model = 'total'
adv = Adversaries(adjacency, F, adv_model, display_flag)
adv.adversary_assignments(states)
print("Adversary Indices: {}".format(adv.adv_indices))

### Test Adversaries: F-local Model
adv_model = 'local'
adv = Adversaries(adjacency, F, adv_model, display_flag)
adv.adversary_assignments(states)
print("Adversary Indices: {}".format(adv.adv_indices))