import sys
import random
import numpy as np 

sys.path.insert(0, "../modules")

from topology_generation import Topology
from objective_functions import DecentralizedQuadratic, DecentralizedDataset
from adversaries import Adversaries
from resilient_algorithms import ResilientAlgorithms
from algorithmic_framework import DistributedAlgorithmFramework


seed = 0
random.seed(seed)
np.random.seed(seed)
np.set_printoptions(precision=2)

### Parameters Set-up for Simulation
# fundamental parameters
func_name = 'banknote'  # chosen from ['quadratic', 'banknote']
alg_names = ['SDMMFD', 'SDFD', 'CWTM', 'CWMed', 'RVO']
n_nodes = 35 if func_name == 'quadratic' else 50
n_dims = 2  # in case of 'quadratic'
n_steps = 10
F = 2

# topology parameters
topo_type = 'r-robust'
period = 1
robust = (2 * n_dims + 1) * F + 1 if func_name == 'quadratic' else (11 * F) + 1
hyperparams_dict = {'threshold': 0.20, 'robustness': robust}

# adversary parameters
adv_model = 'local'

# function parameters
quadratic_type = 'Q-general'  # in case of 'quadratic'
bias_flag = False  # in case of 'banknote'

# framework parameters
alpha = 0.1 if func_name == 'quadratic' else 5e-5
state_inits = {'main': {'label': 'random', 'param': 5.0}, 
               'aux': {'label': 'random', 'param': 5.0}}
stepsize_init = {'type': 'constant', 'param': alpha}

# miscellaneous parameters
n_rand = None
self_flag = True
verify_flag = True
eval_flag = True
display_flag = False


### Initialize Objects
topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
adversaries = Adversaries(topology.adjacencies[0], F, adv_model, display_flag)
if func_name == 'quadratic':
    function = DecentralizedQuadratic(n_nodes, n_dims, quadratic_type)
else:
    function = DecentralizedDataset('banknote', n_nodes, bias_flag, n_rand, display_flag)
    alg_names.remove('RVO')
framework = DistributedAlgorithmFramework(topology, function, adversaries, eval_flag)

### Preliminary Checking
print("\n========== Preliminary Checking ==========")
print("\nNetwork Topology: \n{}".format(topology.adjacencies[0]))
print("\nAdversary Indices: {}".format(adversaries.adv_indices))
print("\nRegular Indices: {}".format(framework.reg_indices))
print("\nNo. of Regular Indices: {}".format(len(framework.reg_indices)))
print("\nLocal Minimizers: \n{}".format(function.minimizers))
print("\nGlobal Minimizer: {}".format(framework.g_minimizer))
print("\nBenchmarks: \n{}".format(framework.benchmarks))


### Execute Resilient Algorithms
for alg_name in alg_names:
    print("\n========== Execution Checking of {} ==========".format(alg_name))
    algorithm = ResilientAlgorithms(alg_name, F, self_flag, verify_flag, display_flag)
    framework.initialization(n_steps, state_inits, stepsize_init)
    framework.distributed_algorithm(algorithm)
    
    # show results
    print("Final Main States: \n{}".format(framework.main3dim[-1]))
    opt_gaps = ["%.2f"%item for item in framework.optimality_gaps]
    print("Optimality Gaps: {}".format(opt_gaps))
    if framework.accracies is not None:
        test_accs = ["%.2f"%item for item in framework.accracies['test']]
        print("Test Accuracies: {}".format(test_accs))
    

