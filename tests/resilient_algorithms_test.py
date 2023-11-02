import sys
import random
import numpy as np 

sys.path.insert(0, "../modules")
from resilient_algorithms import ResilientAlgorithms


np.random.seed(1)

# Parameters Set-up
n_points = 20
n_dims = 2
F = 2
self_flag = True

# Random States Construction
main_self = np.random.randn(n_dims)
main_states = np.random.randn(n_points, n_dims)
aux_self_temp = np.random.randn(n_dims)
aux_states_temp = np.random.randn(n_points, n_dims)

# Default Parameters Set-up
alg_names = ['SDMMFD', 'SDFD', 'CWTM', 'CWMed', 'RVO']
verify_flag = True
display_flag = True

for alg_name in alg_names:
    print("\n\n===== Resilient Algorithm Test: {} =====".format(alg_name))
    if alg_name in ['SDMMFD', 'SDFD']:
        aux_self = aux_self_temp.copy()
        aux_states = aux_states_temp.copy()
    else:
        aux_self = None
        aux_states = None
    # Calculation
    algorithm = ResilientAlgorithms(alg_name, F, self_flag, verify_flag, display_flag)
    main_state, aux_state = algorithm.one_step_computation(main_self, main_states,
                                                           aux_self, aux_states)