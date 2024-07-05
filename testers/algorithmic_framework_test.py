import sys
import random
import numpy as np 

sys.path.insert(0, "modules")

from topology_generation import Topology
from objective_functions import DecentralizedQuadratic, DecentralizedBankNotes
from adversaries import Adversaries
from resilient_algorithms import ResilientAlgorithms
from algorithmic_framework import DistributedAlgorithmFramework


seed = 0
random.seed(seed)
np.random.seed(seed)
np.set_printoptions(precision=4)


# ========== Test: local_observation ==========

n_nodes = 8
n_dims = 2
self_idx = 0
states = np.random.randn(n_nodes, n_dims)
in_indicators = np.random.randint(2, size=n_nodes)
in_indicators[self_idx] = 1

print("\n========== Test: local_observation ==========")
print(f"original states: \n{states}")
print(f"in-neighbor indicators: {in_indicators}")

# construct adv_dict
adv_dict = dict()
for idx in range(n_nodes):
    if idx != self_idx and in_indicators[idx] == 1 and np.random.rand() < 0.5:
        send_vector = np.random.randn(n_dims)
        adv_dict[idx] = {self_idx: send_vector}
        print(f"idx_send: {idx}; vector: {send_vector}")

# calculation
self_state, observed_states = DistributedAlgorithmFramework.local_observation(self_idx, states, in_indicators, adv_dict)
print(f"self idx {self_idx}; self state: {self_state}")
print(f"observed states: \n{observed_states}")


# ========== Test: distributed_algorithm ==========

### Parameters Set-up for Simulation (select "func_name", "attack_info")
# fundamental parameters
func_name = 'banknote'  # chosen from ['quadratic', 'banknote']

alg_names = ['DGD', 'SDMMFD', 'SDFD', 'CWTM', 'CWMed', 'RVO']  # default: ['DGD', 'SDMMFD', 'SDFD', 'CWTM', 'CWMed', 'RVO']
n_nodes = 35 if func_name == 'quadratic' else 50
n_dims = 2  # in case of 'quadratic'
n_steps = 20  # default: 20
F = 2

# topology parameters
topo_type = 'r-robust'
period = 1
robust = (2 * n_dims + 1) * F + 1 if func_name == 'quadratic' else (11 * F) + 1
hyperparams_dict = {'threshold': 0.20, 'robustness': robust}

# adversary parameters
adv_model = 'local'
# attack_info = {'type': 'random', 'param': 20}
# attack_info = {'type': 'label_change', 'param': 0.95}
attack_info = {'type': 'perturbation', 'param': 25}

# function parameters
quadratic_type = 'Q-general'  # in case of 'quadratic'
bias_flag = False  # in case of 'banknote'

# framework parameters
alpha = 0.1 if func_name == 'quadratic' else 5e-5
state_inits = {'main': {'label': 'random', 'param': 5.0}, 
               'aux': {'label': 'random', 'param': 5.0}}
stepsize_init = {'type': 'constant', 'param': alpha}
path_name = 'results/test_run/'
eval_info = {'n_nodes': None, 'period': 2, 'n_eval_iters': None, 'path_name': path_name}

# miscellaneous parameters
n_rand = None
self_flag = True
verify_flag = True
display_flag = False


### Initialize Objects
topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
adversaries = Adversaries(topology.adjacencies[0], F, adv_model, attack_info, display_flag)
if func_name == 'quadratic':
    function = DecentralizedQuadratic(n_nodes, n_dims, quadratic_type)
else:
    function = DecentralizedBankNotes(n_nodes, bias_flag, n_rand, display_flag)
    if 'RVO' in alg_names: alg_names.remove('RVO')
framework = DistributedAlgorithmFramework(topology, function, adversaries, eval_info)


### Preliminary Checking
print("\n\n========== Test: distributed_algorithm ==========")
print("\n========== Preliminary Checking ==========")
print(f"\nNetwork Topology: \n{topology.adjacencies[0]}")
print(f"\nAdversary Indices: {adversaries.adv_indices}")
print(f"\nRegular Indices: {framework.reg_indices}")
print(f"\nNo. of Regular Indices: {len(framework.reg_indices)}")
print(f"\nLocal Minimizers: \n{function.minimizers}")
print(f"\nGlobal Minimizer: {framework.g_minimizer}")
print(f"\nBenchmarks: \n{framework.benchmarks}")

print("\nShow Adversairies' Out-Neighbor Indices (including only regular nodes)")
_ = adversaries.get_out_neighbors_of_adv(adv_flag=False, print_flag=True)

### Execute Resilient Algorithms
for alg_name in alg_names:
    print(f"\n========== Execution Checking of {alg_name} ==========")
    algorithm = ResilientAlgorithms(alg_name, F, self_flag, verify_flag, display_flag)
    framework.initialization(n_steps, state_inits, stepsize_init)
    initial_main2dim = framework.main2dim.copy()
    framework.distributed_algorithm(algorithm)
    
    # show results
    print(f"Initial Main States: \n{initial_main2dim}")
    print(f"Final Main States: \n{framework.main2dim}")
    opt_gaps = [f"{item:.2f}" for item in framework.metric_dict['opt_gap']]
    print(f"Optimality Gaps: {opt_gaps}")
    if framework.metric_dict['test_acc'] is not None:
        test_accs = [f"{item:.2f}%" for item in framework.metric_dict['test_acc']]
        test_accs_w = [f"{item:.2f}%" for item in framework.metric_dict['test_acc_w']]
        print(f"Test Accuracies: {test_accs}")
        print(f"Test Accuracies (worst): {test_accs_w}")