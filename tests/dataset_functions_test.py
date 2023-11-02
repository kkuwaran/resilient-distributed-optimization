import sys
import numpy as np 

sys.path.insert(0, "../modules")
from objective_functions import DecentralizedDataset

### Test All Functions: Decentralized_BankNote Class
print("===== Test All Functions: Decentralized_BankNote Class =====")
np.random.seed(0)
np.set_printoptions(precision=4)

# Parameters Set-up
dataset = 'banknote'
n_nodes = 8
bias_flag = True
n_rand = None
display_flag = True
print("Total Number of Nodes:", n_nodes)
# Calculation
banknote = DecentralizedDataset(dataset, n_nodes, bias_flag, n_rand, display_flag)


print("\n===== Test Function: global_optimal_calculation =====")
# Parameters Set-up
indices = [1, 3, 4, 7]
attribute_flag = True
# Calculation
banknote.global_optimal_calculation(indices, attribute_flag)


print("\n===== Test Function: function_eval =====")
# Parameters Set-up
states = np.random.randn(len(indices), banknote.n_dims)
print("Input States of all Nodes: \n", states)
# Calculation
local_func_vals, global_func_vals, gradients = banknote.function_eval(indices, states)
print("Local Function Value of all Nodes: \n", local_func_vals)
print("Global Function Value of all Nodes: \n", global_func_vals)
print("Gradient of all Nodes: \n", gradients)


print("\n===== Test Function: models_evaluation =====")
# Calculation (training dataset)
print("*** Training Set Results ***")
banknote.models_evaluation(indices, states, 'train')
# Calculation (test dataset)
print("*** Test Set Results ***")
banknote.models_evaluation(indices, states, 'test')