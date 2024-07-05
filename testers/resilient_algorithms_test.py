import sys

import time
import random
import numpy as np 

sys.path.insert(0, "modules")
from resilient_algorithms import ResilientAlgorithms


np.random.seed(2)


# Parameters Set-up
alg_name = 'DGD'
F = 2
self_flag = True
verify_flag = True
display_flag = False
algorithm = ResilientAlgorithms(alg_name, F, self_flag, verify_flag, display_flag)

# ========== Test: one_dimension_filter ==========

n_points = 10
values = np.random.randn(n_points)
direction = 'both'
self_value = np.random.randn()

print("\n========== Test: one_dimension_filter ==========")
print(f"all values {values}; self_value {self_value}")
def one_dimension_filter_test(acceleration_flag):
    algorithm.acceleration_flag = acceleration_flag
    start = time.time()
    filt_indicators = algorithm.one_dimension_filter(values, F, direction, self_value)
    print(f"filtered indicators: {filt_indicators}")
    print(f"acceleration: {algorithm.acceleration_flag}; time: {time.time() - start}")

# test
one_dimension_filter_test(False)
one_dimension_filter_test(True)


# ========== Test: distance_filter ==========

n_points = 5
n_dims = 2
states = np.random.randn(n_points, n_dims)
center = np.random.randn(n_dims)
print("\n========== Test: distance_filter ==========")
print(f"states: \n{states}")
print(f"center: {center}")
# manual check
diffs = states - np.expand_dims(center, axis=0)
distances = np.sqrt(np.sum(diffs ** 2, axis=-1))
print(f"distances (manual check): {distances}")
# function's output
filt_indicators = algorithm.distance_filter(states, center, F, self_state=None)
print(f"filtered indicators: {filt_indicators}")


# ========== Test: minmax_filter ==========

n_points = 10
n_dims = 2
decimals = 2
states = np.round(np.random.randn(n_points, n_dims), decimals)
self_state = np.round(np.random.randn(n_dims), decimals)

print("\n========== Test: minmax_filter ==========")
print(f"states: \n{states}")
print(f"self_state: {self_state}")
def minmax_filter_test(dim_wise_flag, acceleration_flag):
    algorithm.acceleration_flag = acceleration_flag
    start = time.time()
    filt_indicators = algorithm.minmax_filter(states, F, self_state, dim_wise_flag)
    print(f"filtered indicators (dim-wise {dim_wise_flag}): \n{filt_indicators}")
    print(f"acceleration: {algorithm.acceleration_flag}; time: {time.time() - start}")

# per dimension test
minmax_filter_test(True, False)
minmax_filter_test(True, True)
# combined dimension test
minmax_filter_test(False, False)
minmax_filter_test(False, True)


# ========== Test: weighted_average ==========

n_points = 8
n_dims = 2

print("\n========== Test: weighted_average ==========")
def weighted_average_test(states, acceleration_flag):
    print(f"input_state: \n{states}")
    algorithm.acceleration_flag = acceleration_flag
    start = time.time()
    output_state = algorithm.weighted_average(states)
    print(f"output_state: {output_state}")
    print(f"acceleration: {algorithm.acceleration_flag}; time: {time.time() - start}")

# Important Note: there is stochasticity in the output due to the random selection of the weights
# uni-dimension test
states = np.random.randn(n_points)
weighted_average_test(states, False)
weighted_average_test(states, True)

# multi-dimension test
states = np.random.randn(n_points, n_dims)
weighted_average_test(states, False)
weighted_average_test(states, True)


# ========== Test: one_step_computation ==========

# Parameters Set-up
n_points = 12
n_dims = 2
F = 2
self_flag = True

# Random States Construction
main_self = np.random.randn(n_dims)
main_states = np.random.randn(n_points, n_dims)
aux_self_temp = np.random.randn(n_dims)
aux_states_temp = np.random.randn(n_points, n_dims)

# Default Parameters Set-up
alg_names = ['DGD', 'SDMMFD', 'SDFD', 'CWTM', 'CWMed', 'RVO']
acceleration_flag = True
verify_flag = True
display_flag = False

print("\n========== Test: one_step_computation ==========")
for alg_name in alg_names:
    print(f"\n\n===== Resilient Algorithm Test: {alg_name} =====")
    if alg_name in ['SDMMFD', 'SDFD']:
        aux_self = aux_self_temp.copy()
        aux_states = aux_states_temp.copy()
    else:
        aux_self = None
        aux_states = None
    
    # Calculation
    algorithm = ResilientAlgorithms(alg_name, F, self_flag, acceleration_flag, verify_flag, display_flag)
    main_state, aux_state = algorithm.one_step_computation(main_self, main_states, aux_self, aux_states)
    print(f"main_state: {main_state}")
    print(f"aux_state: {aux_state}")