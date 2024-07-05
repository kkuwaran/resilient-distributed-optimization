### Import General Python Modules
import numpy as np
from numba import njit, prange
from scipy.spatial import distance

### Import Private Functions 
from utilities.centerpoint import Point, Centerpoint


@njit
def one_dimension_filter_numba(values, F, direction, self_value=None):
    '''sort one-dimensional values and compute filter indicators (see one_dimension_filter() for details)'''

    n_points = values.size
    filt_indicators = np.zeros(n_points, dtype=np.int32)
    sort_indices = np.argsort(values)
    sort_values = values[sort_indices]
    
    # filter lowest values
    if direction in ['min', 'both']:
        self_val = sort_values[-1] + 1.0 if self_value is None else self_value
        for i in prange(F):
            if sort_values[i] < self_val:
                filt_indicators[sort_indices[i]] = 1
    
    # filter highest values
    if direction in ['max', 'both']:
        self_val = sort_values[0] - 1.0 if self_value is None else self_value
        for i in prange(1, F+1):
            if sort_values[-i] > self_val:
                filt_indicators[sort_indices[-i]] = 1
    
    return filt_indicators


@njit
def minmax_filter_numba(states, F, self_state=None, dim_wise_flag=True):
    '''implement min-max filter (see minmax_filter() for details)'''

    n_points, n_dims = states.shape
    indicators = np.zeros((n_dims, n_points), dtype=np.int32)

    for dim_idx in prange(n_dims):
        scalar_states = states[:, dim_idx]
        self_value = self_state[dim_idx] if self_state is not None else None
        
        sort_indices = np.argsort(scalar_states)
        sort_values = scalar_states[sort_indices]
        
        # filter lowest values
        for i in range(F):
            if self_value is None or sort_values[i] < self_value:
                indicators[dim_idx, sort_indices[i]] = 1
        
        # filter highest values
        for i in range(1, F + 1):
            if self_value is None or sort_values[-i] > self_value:
                indicators[dim_idx, sort_indices[-i]] = 1
    
    # aggregate indicators if dim_wise_flag is False
    if not dim_wise_flag:
        filt_indicators = np.zeros((1, n_points), dtype=np.int32)
        for i in range(n_points):
            if np.any(indicators[:, i]):
                filt_indicators[0, i] = 1
    else:
        filt_indicators = indicators
    return filt_indicators


@njit
def weighted_average_numba(states, weights=None):
    '''compute weighted average of given states (see weighted_average() for details)'''

    n_points = states.shape[0]
    if weights is None:
        randoms = np.random.rand(n_points)
        weights = randoms / np.sum(randoms)
    
    if states.ndim == 1:
        output_state = np.sum(states * weights)
    else:
        output_state = np.sum(states * weights[:, np.newaxis], axis=0)
    
    return output_state


@njit
def CWTM_computation_numba(main_states, self_state, self_flag, F):
    '''execute one step of CWTM algorithm optimized with Numba (see CWTM_computation() for details)'''

    n_points, n_dims = main_states.shape
    filt_self_state = self_state.copy() if self_flag else None
    main_state = np.zeros(n_dims)
    
    # filter function already optimized with Numba
    mm_indicators = minmax_filter_numba(main_states, F, filt_self_state, True)
    
    for dim_idx in range(n_dims):
        node_indicator = mm_indicators[dim_idx] == 0
        rem_values = main_states[node_indicator, dim_idx]
        rem_self_values = np.empty(rem_values.shape[0] + 1, dtype=rem_values.dtype)
        rem_self_values[0] = self_state[dim_idx]
        rem_self_values[1:] = rem_values
        main_state[dim_idx] = weighted_average_numba(rem_self_values, None)
        
    return main_state


class ResilientAlgorithms:

    # algorithm names (whether auxiliary states are used or not)
    ALG_NAMES1 = ['DGD', 'CWTM', 'CWMed', 'RVO']
    ALG_NAMES2 = ['SDMMFD', 'SDFD', 'R-SDMMFD']

    def __init__(self, alg_name, F, self_flag=True, acceleration_flag=False, verify_flag=False, display_flag=False):
        '''Calculate one step of specified resilient algorithm using one_step_computation()
        ========== Inputs ==========
        alg_name - string: resilient algorithm chosen from ['DGD', 'SDMMFD', 'SDFD', 'R-SDMMFD', 'CWTM', 'CWMed', 'RVO']
        F - nonnegative integer: number of nodes to be filtered in a direction
        acceleration_flag - True/False: flag determining whether to use acceleration
        self_flag - True/False: flag determining whether including self state in the filter
        verify_flag - True/False: execute inputs_verification()
        display_flag - True/False: show information
        '''
        
        # fundamental attributes
        assert alg_name in self.ALG_NAMES1 + self.ALG_NAMES2, f'incorrect algorithm name: {alg_name}'
        self.alg_name = alg_name
        self.F = F
        self.self_flag = self_flag
        self.aux_flag = alg_name in self.ALG_NAMES2

        # auxiliary attributes
        self.acceleration_flag = acceleration_flag
        self.verify_flag = verify_flag
        self.display_flag = display_flag
        if display_flag: np.set_printoptions(precision=3)

        
    # ============================== Algorithm Component Functions ==============================
    
    def one_dimension_filter(self, values, F, direction, self_value=None):
        '''sort one-dimensional values and compute filter indicators 
        ========== Inputs ==========
        values - ndarray (n_points, ): values to be filtered
        F - nonnegative integer: number of nodes to be filtered in a direction
        direction - string: filter direction chosen from ['min', 'max', 'both']
        self_value - scalar or None: value of the operating node
        ========== Outputs ==========
        filt_indicators - list of len=n_points: point indicator with 1:Filtered, 0:Survive
        '''

        if self.acceleration_flag:
            filt_indicators = one_dimension_filter_numba(values, F, direction, self_value)
        else:
            n_points = values.size
            filt_indicators = [0] * n_points
            sort_indices = np.argsort(values)
            sort_values = values[sort_indices]
            
            # filter lowest values
            if direction in ['min', 'both']:
                # when self_value is None, set self_val not to disturb the filter
                self_val = sort_values[-1] + 1.0 if self_value is None else self_value
                for i in range(F):
                    if sort_values[i] < self_val:
                        filt_indicators[sort_indices[i]] = 1
            
            # filter highest values
            if direction in ['max', 'both']:
                # when self_value is None, set self_val not to disturb the filter
                self_val = sort_values[0] - 1.0 if self_value is None else self_value
                for i in range(1, F+1):
                    if sort_values[-i] > self_val:
                        filt_indicators[sort_indices[-i]] = 1
        
        return filt_indicators
    
    
    def distance_filter(self, states, center, F, self_state=None):
        '''Implement Distance Filter w.r.t. center
        ========== Inputs ==========
        states - ndarray (n_points, n_dims): states
        center - ndarray (n_dims, ): vector at the center of filter
        F - nonnegative integer: number of nodes to be filtered
        self_state - ndarray (n_dims, ) or None: state of the operating node
        ========== Outputs ==========
        filt_indicators - list of len=n_points: point indicator with 1:Filtered, 0:Survive
        '''
        
        n_points = states.shape[0]
        assert n_points > F, "not enough points (distance filter)"
        
        # compute distance from states to center
        center_temp = np.expand_dims(center, axis=0)
        distances = distance.cdist(center_temp, states, 'euclidean')[0] 
        # compute distance from self_state to center
        self_distance = np.linalg.norm(self_state - center) if self_state is not None else None
        
        # filter out highest distances
        filt_indicators = self.one_dimension_filter(distances, F, 'max', self_distance)
        
        if self.display_flag:
            print("\n===== Distance Filter Information =====")
            print(f"Input States: \n{states}")
            print(f"Filter Parameter (F): {F}; Center: {center}")
            print(f"Filter Indicators (result): {filt_indicators}")
        
        return filt_indicators
        
    
    def minmax_filter(self, states, F, self_state=None, dim_wise_flag=True):
        '''Implement Min-Max Filter 
        ========== Inputs ==========
        states - ndarray (n_points, n_dims): states
        F - nonnegative integer: number of nodes to be filtered
        self_state - ndarray (n_dims, ) or None: state of the operating node
        dim_wise_flag - True/False: if True, filter dimension-wise; otherwise, filter all dimensions together
        ========== Outputs ==========
        filt_indicators - list of len=n_points or list of lists (n_dims, n_points): 
                          point indicator with 1:Filtered, 0:Survive
        '''
    
        n_points, n_dims = states.shape
        if dim_wise_flag: 
            assert n_points > 2 * F, "not enough points (min-max filter)"
        else:
            assert n_points > 2 * n_dims * F, "not enough points (min-max filter)"

        if self.acceleration_flag:
            filt_indicators = minmax_filter_numba(states, F, self_state, dim_wise_flag)
            if not dim_wise_flag:
                filt_indicators = filt_indicators.reshape(-1)
        else:
            # calculate indicator for each dimension
            indicators = []
            for dim_idx in range(n_dims):
                # get values in a given dimension
                scalar_states = states[:, dim_idx]
                self_value = self_state[dim_idx] if self_state is not None else None
                # filter out lowest and highest values
                indicator = self.one_dimension_filter(scalar_states, F, 'both', self_value)
                indicators.append(indicator)
            
            if not dim_wise_flag:
                filt_indicators = [0] * n_points
                # any dim of a point is 1 -> filtered
                for i in range(n_points):
                    aggregated_dims = [indicators[dim_idx][i] for dim_idx in range(n_dims)]
                    filt_indicators[i] = int(any(aggregated_dims))
            else:
                filt_indicators = indicators
            
        if self.display_flag:
            print("\n===== Min-Max Filter Information =====")
            print(f"Input States: \n{states}")
            print(f"Filter Parameter (F): {F}; Dim-Wise Flag: {dim_wise_flag}")
            print(f"Filter Indicators (result): \n{filt_indicators}")
            
        return filt_indicators


    def weighted_average(self, states, weights=None):
        '''Compute Weighted Average of given states
        ========== Inputs ==========
        states - ndarray (n_points, ) or (n_points, n_dims): states
        weights - ndarray (n_points, ): weight for each state
        ========== Outputs ==========
        output_state - scalar or ndarray (n_dims, ): output state
        ========== Notes ==========
        if weights is None, then randomized weights are generated for computing convex combination
        '''

        if self.acceleration_flag:
            output_state = weighted_average_numba(states, weights)
        else:
            # construct randomized weights
            n_points = states.shape[0]
            if weights is None:
                randoms = np.random.rand(n_points)
                weights = randoms / np.sum(randoms)
            
            # compute weighted average
            if states.ndim == 2:
                weights = np.expand_dims(weights, axis=1)
            output_state = np.sum(states * weights, axis=0, keepdims=False)
        
        if self.display_flag:
            print("\n===== Weighted Average Information =====")
            print(f"Input States: \n{states}")
            print(f"Input Weights: {weights}")
            print(f"Output State: \n{output_state}")
            
        return output_state


    # ============================== Algorithm Functions ==============================

    def DGD_computation(self, main_states, self_state):
        '''Execute One Step of DGD Algorithm (without gradient step)
        ========== Inputs ==========
        main_states - ndarray (n_points, n_dims): in-neighbors' states
        self_state - ndarray (n_dims, ): state of the operating node
        ========== Outputs ==========
        main_state - ndarray (n_dims, ): main state output
        aux_state - None
        '''

        aug_main_states = np.insert(main_states.copy(), 0, self_state.copy(), axis=0)  
        main_state = self.weighted_average(aug_main_states, weights=None)
        return main_state, None
    
        
    def SDXXFD_computation(self, main_states, aux_states, self_state, center, mm_flag):
        '''Execute One Step of SDMMFD or SDFD Algorithm
        ========== Inputs ==========
        main_states - ndarray (n_points, n_dims): main states to be filtered
        aux_states - ndarray (n_points, n_dims): auxiliary states to be filtered
        self_state - ndarray (n_dims, ): main states of the operating node
        center - ndarray (n_dims, ): vector at the center of distance filter
        mm_flag - True/False: if True, run SDMMFD; if False, run SDFD
        ========== Outputs ==========
        main_state - ndarray (n_dims, ): main state output
        aux_state - ndarray (n_dims, ): auxiliary state output
        '''
        
        # fundamental parameters
        self_main = self_state.copy() if self.self_flag else None
            
        # main states computation
        dist_indicators = self.distance_filter(main_states, center, self.F, self_main)
        rem_main_states = main_states[np.array(dist_indicators) == 0]
        if mm_flag:
            mm_indicators = self.minmax_filter(rem_main_states, self.F, self_main, False)
            rem_main_states = rem_main_states[np.array(mm_indicators) == 0]
        main_state, _ = self.DGD_computation(rem_main_states, self_state)
        
        # auxiliary states computation
        aux_state, _ = self.CWTM_computation(aux_states, center)
        return main_state, aux_state
    

    def RSDMMFD_computation(self, main_states, aux_states, self_state, center):
        '''Execute One Step of RSDMMFD Algorithm
        RSDMMFD: reduced redundancy requirement version of SDMMFD
        ========== Inputs / Outputs ==========
        see SDXXFD_computation() for details
        '''

        # fundamental parameters
        self_main = self_state.copy() if self.self_flag else None

        # main states computation
        dist_indicators = self.distance_filter(main_states, center, self.F, self_main)
        rem_main_states = main_states[np.array(dist_indicators) == 0]
        main_state, _ = self.CWTM_computation(rem_main_states, self_state)

        # auxiliary states computation
        aux_state, _ = self.CWTM_computation(aux_states, center)
        return main_state, aux_state

    
    def CWTM_computation(self, main_states, self_state):
        '''Execute One Step of CWTM Algorithm
        ========== Inputs ==========
        main_states - ndarray (n_points, n_dims): main states to be filtered
        self_state - ndarray (n_dims, ): state of the operating node
        ========== Outputs ==========
        main_state - ndarray (n_dims, ): main state output
        aux_state - None
        '''

        if self.acceleration_flag:
            main_state = CWTM_computation_numba(main_states, self_state, self.self_flag, self.F)
        else:
            # fundamental parameters and storage
            filt_self_state = self_state.copy() if self.self_flag else None
            n_dims = main_states.shape[1]
            main_state = np.zeros(n_dims)
        
            # main states computation
            mm_indicators = self.minmax_filter(main_states, self.F, filt_self_state, True)
            for dim_idx in range(n_dims):
                node_indicator = np.array(mm_indicators[dim_idx]) == 0
                rem_values = main_states[node_indicator, dim_idx]
                rem_self_values = np.insert(rem_values, 0, self_state[dim_idx], axis=0)
                main_state[dim_idx] = self.weighted_average(rem_self_values, weights=None)

        return main_state, None
    
    
    def CWMed_computation(self, main_states, self_state):
        '''Execute One Step of CWMed Algorithm
        ========== Inputs ==========
        main_states - ndarray (n_points, n_dims): main states to be calculated
        self_state - ndarray (n_dims, ): state of the operating node
        ========== Outputs ==========
        main_state - ndarray (n_dims, ): main state output
        aux_state - None
        '''
        
        # calculate median of neighbors' states
        main_state = np.median(main_states, axis=0)
        # (randomized) convex combination with self_state
        temp_states = np.stack([self_state, main_state], axis=0)
        main_state = self.weighted_average(temp_states)
        return main_state, None
    
    
    def RVO_computation(self, main_states, self_state):
        '''Execute One Step of RVO Algorithm
        ========== Inputs ==========
        main_states - ndarray (n_points, n_dims): main states to be filtered
        self_state - ndarray (n_dims, ): state of the operating node
        ========== Outputs ==========
        main_state - ndarray (n_dims, ): main state output
        aux_state - None
        '''
        
        assert main_states.shape[1] == 2, "incorrect dimension for RVO"
        # calculate centerpoint
        points = []
        points = [Point(state[0], state[1]) for state in main_states]
        cp = Centerpoint(plot=False)
        centerpoint = cp.reduce_then_get_centerpoint(points)
        main_state = np.array([centerpoint.x, centerpoint.y])
        
        # (randomized) convex combination with self_state
        temp_states = np.stack([self_state, main_state], axis=0)
        main_state = self.weighted_average(temp_states)
        return main_state, None
        
        
    # ============================== [Main] Algorithm Call ==============================
    
    def inputs_verification(self, main_self_state, main_neighbor_states, 
                            aux_self_state, aux_neighbor_states):
        '''Check Validity of Inputs of one_step_computation()'''
        
        # check type and dimension of main states and auxiliary states
        assert isinstance(main_self_state, np.ndarray) and main_self_state.ndim == 1, "incorrect main self state type or dimension"
        assert isinstance(main_neighbor_states, np.ndarray) and main_neighbor_states.ndim == 2, "incorrect main neighbor states type or dimension"
        if self.aux_flag:
            assert aux_self_state.shape == main_self_state.shape, "incorrect auxiliary self state type or dimension"
            assert aux_neighbor_states.shape == main_neighbor_states.shape, "incorrect auxiliary neighbor states type or dimension"
        else:
            assert aux_self_state is None, "auxiliary self state must be None"
            assert aux_neighbor_states is None, "auxiliary neighbor states must be None"
            if self.alg_name == 'RVO':
                assert main_self_state.shape[0] == 2, "RVO requires states dimension = 2"
                assert main_neighbor_states.shape[1] == 2, "RVO requires states dimension = 2"
            
        
    def one_step_computation(self, main_self_state, main_neighbor_states, 
                             aux_self_state=None, aux_neighbor_states=None):
        '''Execute One Step of a Resilient Algorithm
        ========== Inputs ==========
        main_self_state - (n_dims, ): main state of self
        main_neighbor_states - ndarray (n_points, n_dims): main states of in-neighbors
        aux_self_state - (n_dims, ) or None: auxiliary state of self
        aux_neighbor_states - ndarray (n_points, n_dims) or None: auxiliary states of in-neighbors
        '''
  
        # verify inputs
        if self.verify_flag:
            self.inputs_verification(main_self_state, main_neighbor_states,
                                     aux_self_state, aux_neighbor_states)

        # execute algorithm
        if self.alg_name == 'DGD':
            main_state, aux_state = self.DGD_computation(main_neighbor_states, main_self_state)
        elif self.alg_name == 'SDMMFD':
            main_state, aux_state = self.SDXXFD_computation(main_neighbor_states, aux_neighbor_states, 
                                                            main_self_state, aux_self_state, True)
        elif self.alg_name == 'SDFD':
            main_state, aux_state = self.SDXXFD_computation(main_neighbor_states, aux_neighbor_states, 
                                                            main_self_state, aux_self_state, False)
        elif self.alg_name == 'R-SDMMFD':
            main_state, aux_state = self.RSDMMFD_computation(main_neighbor_states, aux_neighbor_states, 
                                                            main_self_state, aux_self_state)
        elif self.alg_name == 'CWTM':
            main_state, aux_state = self.CWTM_computation(main_neighbor_states, main_self_state)
        elif self.alg_name == 'CWMed':
            main_state, aux_state = self.CWMed_computation(main_neighbor_states, main_self_state)
        elif self.alg_name == 'RVO':
            main_state, aux_state = self.RVO_computation(main_neighbor_states, main_self_state)
        else:
            raise ValueError("incorrect algorithm name")
            
        # display information
        if self.display_flag:
            print(f"\n===== {self.alg_name} Computation Information =====")
            print(f"Input Main States: \n{main_neighbor_states}")
            if self.aux_flag:
                print(f"Input Auxiliary States: \n{aux_neighbor_states}")
            print(f"Self State: {main_self_state}")
            print(f"Output Main State: {main_state}")
            print(f"Output Auxiliary State: {aux_state}")
        
        return main_state, aux_state  