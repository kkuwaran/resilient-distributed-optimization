import sys
import numpy as np 

sys.path.insert(0, "modules")
from topology_generation import Topology

def show_results(adjacencies, stoch_type):
    # retrieve parameters
    period = adjacencies.shape[0]
    n_nodes = adjacencies.shape[-1]
    # show adjacency matrices
    print("Adjacency Matrices Collection: \n", adjacencies)
    # calculate stochastic matrices
    stoch_matrices = np.zeros((period, n_nodes, n_nodes))
    for p in range(period): 
        stoch_matrices[p] = topology.stochastic_matrices_gen(topology.adjacencies[p], stoch_type)
    # show stochastic matrices
    print("Stochastic Matrices Collection: \n", stoch_matrices)
    # calculate parameters
    try:
        delta, beta = topology.parameters(stoch_matrices)
    except AssertionError: 
        delta, beta = None, None
    print("Spectral Gap:", delta, "and Identity Gap:", beta)
    
    

### Test Function: adjacency_matrices_gen and stochastic_matrices_gen
print("===== Test Functions: adjacency_matrices_gen and stochastic_matrices_gen =====")
np.random.seed(0)
np.set_printoptions(precision=2)

print("\n***** Topology: complete_ring *****")
# Parameters Set-up
topo_type = 'complete_ring'
n_nodes = 8
period = 2
stoch_type = 'column'
# Calculation
topology = Topology(topo_type, n_nodes, period)
show_results(topology.adjacencies, stoch_type)

print("\n***** Topology: partial_ring *****")
# Parameters Set-up
topo_type = 'partial_ring'
n_nodes = 8
period = 2
sym_flag = True
stoch_type = 'doubly'
# Calculation
topology = Topology(topo_type, n_nodes, period, sym_flag)
show_results(topology.adjacencies, stoch_type)

print("\n***** Topology: random *****")
# Parameters Set-up
topo_type = 'random'
n_nodes = 8
period = 2
hyperparams_dict = {"threshold": 0.50}
#threshold = 0.50
stoch_type = 'column'
# Calculation
topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
show_results(topology.adjacencies, stoch_type)

print("\n***** Topology: r-robust *****")
# Parameters Set-up
topo_type = 'r-robust'
n_nodes = 8
T_horizon = 4
period = 2
hyperparams_dict = {"threshold": 0.50, "robustness": 3}
stoch_type = 'doubly'
# Calculation
topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
show_results(topology.adjacencies, stoch_type)