import sys
import numpy as np 

sys.path.insert(0, "modules")
from topology_generation import Topology
from adversaries import Adversaries


np.random.seed(0)

# Parameters Set-up for Topology class
topo_type = 'r-robust'
n_nodes = 10
period = 1
hyperparams_dict = {'threshold': 0.20, 'robustness': 2}
# Get adjacency matrix
topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
adjacency = topology.adjacencies[0]


# Parameters Set-up for Adversaries class
F = 3
# attack_info = {'type': 'byzantine', 'param': 10}
# attack_info = {'type': 'label_change', 'param': 0.5}
# attack_info = {'type': 'perturbation', 'param': 3, 'perturbation_mode': 'fixed-norm', 'broadcast_flag': False}
attack_info = {'type': 'perturbation', 'param': 3, 'perturbation_mode': 'gaussian', 'broadcast_flag': True}
n_dims = 2
display_flag = False

# states construction for test
states = np.random.randn(n_nodes, n_dims)
print(f"states of all nodes: \n{states}")


# Function to display the results of the adversaries
def show_results(adv, adv_dict):
    print(f"Adversary Indices: {adv.adv_indices}")
    for idx_send in adv.adv_indices:
        print(f"Adjacency Indicators: {adv.adjacency[:, idx_send]}")
        for idx_receive, adv_state in adv_dict[idx_send].items():
            print(f"AdvNode {idx_send} -> Node {idx_receive}: {adv_state}")

### Test Adversaries: F-total Model
print("\n========== Test: F-total Model ==========")
adv_model = 'total'
adv = Adversaries(adjacency, F, adv_model, attack_info, display_flag)
print(f"Average in-neighbors: {adv.avg_inneighbors}")
adv_dict = adv.adversary_assignments(states)
show_results(adv, adv_dict)
_ = adv.get_out_neighbors_of_adv(adv_flag=False, print_flag=True)

### Test Adversaries: F-local Model
print("\n========== Test: F-local Model ==========")
adv_model = 'local'
adv = Adversaries(adjacency, F, adv_model, attack_info, display_flag)
print(f"Average in-neighbors: {adv.avg_inneighbors}")
adv_dict = adv.adversary_assignments(states)
show_results(adv, adv_dict)
_ = adv.get_out_neighbors_of_adv(adv_flag=False, print_flag=True)