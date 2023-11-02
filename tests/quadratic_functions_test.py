import sys
import numpy as np 

sys.path.insert(0, "../modules")
from objective_functions import DecentralizedQuadratic


### Test All Functions: Decentralized_Quadratic Class
print("===== Test All Functions: Decentralized_Quadratic Class =====\n")
np.random.seed(0)
np.set_printoptions(precision=4)

# Parameters Set-up
n_nodes = 8
n_dims = 4
quadratic_type = 'Q-general'
indices = list(range(n_nodes))
print("Total Number of Nodes:", n_nodes)

# class Initialization
quadratic = DecentralizedQuadratic(n_nodes, n_dims, quadratic_type)
quadratic.global_optimal_calculation(indices)


print("\n===== Test Function: function_eval =====")
states = np.random.randn(n_dims, len(indices)).T
print("Input States of all Nodes: \n", states)
# Calculation
local_func_vals, global_func_vals, gradients = quadratic.function_eval(indices, states)
print("Local Function Values of all Nodes: \n", local_func_vals)
print("Global Function Values of all Nodes: \n", global_func_vals)
print("Gradients of all Nodes: \n", gradients)