import os

import time
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from topology_generation import Topology
from adversaries import Adversaries
from objective_functions_cifar10 import DecentralizedCIFAR10
from resilient_algorithms import ResilientAlgorithms
from algorithmic_framework import DistributedAlgorithmFramework_ML


# variables for CIFAR-10 experiments
seed = 42   
alg_name = 'DGD'  # chosen from ['DGD', 'CWTM', 'SDFD', 'R-SDMMFD']
n_adv = 6   

# fundamental parameters
n_nodes = 50
n_epochs = 350

# function parameters
batch_size = 64
bias_flag = False

# topology parameters
topo_type = 'r-robust'
period = 1
hyperparams_dict = {'threshold': 0.20, 'robustness': 19}
F = 6

# adversary model parameters
attack_info = {'type': 'perturbation', 'param': 0.045, 
               'perturbation_mode': 'gaussian', 'broadcast_flag': False}
adv_model = 'local'

# baseline parameters
baseline_epochs = 30
baseline_optimizer = 'SGD'
baseline_opt_config = {'learning_rate': 0.002, 'momentum': 0.9}

# framework parameters
state_inits = {'main': None, 'aux': None}
stepsize_init = {'type': 'constant', 'param': 0.04}
directory = f'results/exp_cifar10_adv{n_adv}/'
eval_info = {'n_nodes': None, 'period': 5, 'n_eval_iters': 100, 'path_name': directory}

# miscellaneous parameters
self_flag = True
acceleration_flag = True
verify_flag = False
display_flag = False


# ==================== Execute Experiment ====================

### create directory
if not os.path.exists(directory):
    os.makedirs(directory)

### set random seed
random.seed(seed)
np.random.seed(seed)
tf.experimental.numpy.random.seed(seed)


### initialize objects
topology = Topology(topo_type, n_nodes, period, **hyperparams_dict)
adversaries = Adversaries(topology.adjacencies[0], n_adv, adv_model, attack_info, display_flag)
function = DecentralizedCIFAR10(n_nodes, batch_size, bias_flag, adversaries.reg_indices, display_flag)
framework = DistributedAlgorithmFramework_ML(topology, function, adversaries, eval_info)
algorithm = ResilientAlgorithms(alg_name, F, self_flag, acceleration_flag, verify_flag, display_flag)

### set-up initialization and execute algorithm
start_time_main = time.time()
framework.initialization(n_epochs, state_inits, stepsize_init)
framework.distributed_algorithm(algorithm)
print(f"Time taken: {time.time() - start_time_main:.2f}s")

np.set_printoptions(precision=2)
print(f"Test Accuracy: {np.array(framework.metric_dict['test_acc_avg'])}")
print(f"Adversary Indices: {framework.adv_indices}")
print(f"Number of Edges: {int(np.sum(topology.adjacencies[0] / 2))}")

### train baseline
if alg_name == 'DGD':
    start_time_baseline = time.time()
    baseline_dict = function.train_baseline(baseline_epochs, baseline_optimizer, baseline_opt_config, directory)
    print(f"Baseline: Time {time.time() - start_time_baseline:.2f}s; Test Accuracy {baseline_dict['test_acc'][-1]} \n")


# ==================== Plot Results ====================

x = framework.eval_timesteps
names = ['_acc', '_loss']
ylabels = ['Accuracy', 'Loss']
for name, ylabel in zip(names, ylabels):
    # plot average accuracies
    plt.plot(x, framework.metric_dict['train' + name + '_avg'], label='train', color='C0')
    plt.plot(x, framework.metric_dict['test' + name + '_avg'], label='test', color='C1')
    # plot worst accuracies
    plt.plot(x, framework.metric_dict['train' + name + '_w'], label='train (worst)', linestyle='dashed', color='C0')
    plt.plot(x, framework.metric_dict['test' + name + '_w'], label='test (worst)', linestyle='dashed', color='C1')
    plt.title(f"{alg_name} Algorithm; {attack_info['perturbation_mode']} {attack_info['param']}")
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()