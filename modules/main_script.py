import random
import numpy as np 

from topology_generation import Topology
from objective_functions import DecentralizedQuadratic, DecentralizedDataset
from adversaries import Adversaries
from resilient_algorithms import ResilientAlgorithms
from algorithmic_framework import DistributedAlgorithmFramework
from experiments import Experiment


seed = 7
random.seed(seed)
np.random.seed(seed)
np.set_printoptions(precision=2)


### Choose Experiment 
experiment_name = 'banknote'  # chosen from ['quadratic', 'banknote']


### Parameters Configuration
if experiment_name == 'quadratic':
    # fundamental parameters
    alg_names = ['SDMMFD', 'SDFD', 'CWTM', 'RVO']  # name of algorithms (see resilient_algorithms.py)
    n_nodes = 40  # total number of nodes (see topology_generation.py)
    n_dims = 2  # number of independent variables for quadratic functions (see objective_functions.py)
    n_steps = 300  # number of time-steps in simulation (see algorithmic_framework.py)
    n_rounds = 5  # number of times to re-run the experiment (see experiments.py)
    
    # topology parameters and adversary model
    topo_type = 'r-robust'  # type of network topology (see topology_generation.py)
    period = 1  # set to 1 in case of time-invariant topology (see topology_generation.py)
    hyperparams_dict = {'threshold': 0.40, 'robustness': 11}  # threshold: density of additional edges; robustness: the parameter of r-robust topology (see topology_generation.py)
    F = 2  # the parameter of F-local or F-total adversary model (see adversaries.py)
    adv_model = 'local'  # model of adversaries 'local' or 'total' (see adversaries.py)
    
    # function parameters
    quadratic_type = 'Q-general'  # type of generated quadratic functions 'Q-diag' or 'Q-general' (see objective_functions.py)
    
    # framework parameters
    state_inits = {'main': {'label': 'random', 'param': 20.0}, 
                   'aux': {'label': 'minimizer', 'param': None}}  # states initialization (see algorithmic_framework.py)
    stepsize_init = {'type': 'constant', 'param': 0.05}  # step-size initialization (see algorithmic_framework.py)
    
    # plotting parameters
    plt_dict = {'dist_min': None, 'opt_gap': None, 'cons_diam': [None, 100, None, None]}  # axes_limits as values (see experiments.py)
    
    # miscellaneous parameters
    self_flag = True  # whether to consider self state as a special one which cannot be eliminated in the filtering process (see resilient_algorithms.py)
    verify_flag = False  # whether to check the inputs to the algorithm every time-step (see resilient_algorithms.py)
    eval_flag = True  # whether to evaluate the algorithms (see algorithmic_framework.py)
    display_flag = False  # whether to show details (for debugging purpose)
    savefig = True  # whether to save the experimental results (see experiments.py)
    path_name = '../figures/exp_quadratic/'  # path to save the results (see experiments.py)
    
    ### Initialize Objects
    topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
    adversaries = Adversaries(topology.adjacencies[0], F, adv_model, display_flag)
    function = DecentralizedQuadratic(n_nodes, n_dims, quadratic_type)
    framework = DistributedAlgorithmFramework(topology, function, adversaries, eval_flag)
    algorithms = [ResilientAlgorithms(alg_name, F, self_flag, verify_flag, display_flag) for alg_name in alg_names]
    
    ### Run Experiments
    exp = Experiment(framework, algorithms, n_rounds, n_steps, state_inits, stepsize_init)
    for plt_name, axes_limits in plt_dict.items():
        exp.plot_settings(plt_name, axes_limits, savefig, path_name)
    
elif experiment_name == 'banknote':
    # fundamental parameters
    alg_names = ['SDMMFD', 'SDFD', 'CWTM']
    n_nodes = 75
    n_steps = 250
    n_rounds = 6
    
    # topology parameters and adversary model
    F = 2
    adv_model = 'local'
    topo_type = 'r-robust'
    period = 1
    hyperparams_dict = {'threshold': 0.25, 'robustness': 23}
    
    # function parameters
    bias_flag = True  # whether to distribute the data with the same label to each node (see objective_functions.py)
    
    # framework parameters
    state_inits = {'main': {'label': 'random', 'param': 20.0}, 
                   'aux': {'label': 'random', 'param': 20.0}}
    stepsize_init = {'type': 'constant', 'param': 8e-4}
    
    # plotting parameters
    plt_dict = {'dist_min': [0, n_steps, 8, 15], 'opt_gap': [0, n_steps, 1e1, 1e3], 'cons_diam': [0, n_steps, 0.4, 10],
                'train': [0, 100, 50, 100], 'test': [0, 100, 50, 100]}  # axes_limits as values
    
    # miscellaneous parameters
    n_rand = None  # random state for shuffling data (see objective_functions.py)
    self_flag = True
    verify_flag = False
    eval_flag = True
    display_flag = False
    savefig = True
    path_name = '../figures/exp_banknote/'
    
    ### Initialize Objects
    topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
    adversaries = Adversaries(topology.adjacencies[0], F, adv_model, display_flag)
    function = DecentralizedDataset(experiment_name, n_nodes, bias_flag, n_rand, display_flag)
    framework = DistributedAlgorithmFramework(topology, function, adversaries, eval_flag)
    algorithms = [ResilientAlgorithms(alg_name, F, self_flag, verify_flag, display_flag) for alg_name in alg_names]
    
    ### Run Experiments
    exp = Experiment(framework, algorithms, n_rounds, n_steps, state_inits, stepsize_init)
    for plt_name, axes_limits in plt_dict.items():
        exp.plot_settings(plt_name, axes_limits, savefig, path_name)

else:
    NameError("incorrect experiment name")



