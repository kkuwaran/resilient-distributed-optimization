import numpy as np
from scipy.spatial import distance


class DistributedAlgorithmFramework:
    def __init__(self, topology, function, adversaries, eval_flag=True):
        '''Initialize Resilient Distributed Optimization Algorithm Attributes
        ========== Inputs ==========
        topology - Topology class: topology of network (i.e., adjacency matrices)
        function - DecentralizedQuadratic or DecentralizedDataset class:
                    function for all nodes in the network
        adversaries - Adversaries class: location and behavior of adversaries
        eval_flag - True/False: algorithm evaluation
        '''
        
        self.topology = topology
        self.function = function
        self.adv = adversaries
        
        # fundamental attributes
        self.n_nodes = self.topology.n_nodes
        self.n_dims = self.function.n_dims
        self.adv_indices = self.adv.adv_indices  # list of adversary indices
        self.reg_indices = [node for node in range(self.n_nodes) if node not in self.adv_indices]
        
        # global information attributes
        self.function.global_optimal_calculation(self.reg_indices)
        self.g_minimizer = self.function.global_minimizer  # ndarray (n_dims, )
        self.g_opt_value = self.function.global_opt_value  # scalar
        
        # algorithm simulation attributes by initialization()
        self.n_steps = None
        self.adjacencies = None  # ndarray (n_steps, n_nodes, n_nodes)
        self.main3dim = None  # ndarray (n_steps+1, n_nodes, n_dims)
        self.aux3dim = None  # ndarray (n_steps+1, n_nodes, n_dims) or None
        self.stepsizes = None  # ndarray (n_steps, )
        
        # miscellaneous attributes
        self.max_grad_norm = 1e6  # maximum value of gradient norm
        self.show_period = 20  # print current step every period
        
        # algorithm evaluation attributes (determined in states_evaluation())
        self.eval_flag = eval_flag
        self.distances = None
        self.optimality_gaps = None
        self.consensus_diameters = None
        self.accracies = None  # {'train': [], 'worst_train': [], 'test': [], 'worst_test': []}
        self.argmin_accuracy_records = None
        self.benchmarks = None
        
        # calculate benchmarks for each evaluation metric
        if self.eval_flag:
            self.benchmarks_calculation()
    
    
    # ============================== Benchmarks Calculation Functions ==============================
    
    def benchmarks_calculation(self):
        '''Calculate Benchmarks for each Evaluation Metric
        ========== Outputs ==========
        benchmarks['dist_min'] - non-negative scalar: min of minimizers to global minimizer 
                                over regular nodes; min | x_i^* - x^* |
        benchmarks['opt_gap'] - non-negative scalar: min of optimality gap computed at minimizers 
                                over regular nodes; min f(x_i^*) - f^*
        benchmarks['train'] - scalar [0, 100]: training accuracy computed using global minimizer
        benchmarks['test'] - scalar [0, 100]: test accuracy computed using global minimizer
        '''
        
        self.benchmarks = {'dist_min': None, 'opt_gap': None, 'cons_diam': None, 
                           'train': None, 'w_train': None, 'test': None, 'w_test': None}
        
        if self.function.minimizers is not None:
            reg_minimizers = self.function.minimizers[self.reg_indices]
            
            # distance to minimizer
            diffs = reg_minimizers - np.expand_dims(self.g_minimizer, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            min_distance = np.min(distances)
            
            # optimality gap
            _, g_f_vals, _ = self.function.function_eval(self.reg_indices, reg_minimizers)
            gap_vals = g_f_vals - self.g_opt_value
            min_gap = np.min(gap_vals)
            
            # store results
            self.benchmarks['dist_min'] = min_distance
            self.benchmarks['opt_gap'] = min_gap
        
        if hasattr(self.function, 'models_evaluation'):
            state = np.expand_dims(self.g_minimizer, axis=0)
            dummy_indices = [0]
            _, g_train_acc = self.function.models_evaluation(dummy_indices, state, 'train')
            _, g_test_acc = self.function.models_evaluation(dummy_indices, state, 'test')
            
            # store results
            self.benchmarks['train'] = g_train_acc
            self.benchmarks['w_train'] = g_train_acc
            self.benchmarks['test'] = g_test_acc
            self.benchmarks['w_test'] = g_test_acc
        
        
    # ============================== Initialization Functions ==============================
    
    def initialization(self, n_steps, state_inits, stepsize_init):
        '''Initialize Sequence of Adjacency Matrices, Main/Auxiliary States, and Step-size Schedule
        ========== Inputs ==========
        n_steps - positive integer: number of simulation steps
        state_inits - dict of dict: {'main': state_init, 'aux': state_init} where
                                    state_init = {'label': init label, 'param': parameter associated to init}
                                    (see states_initialization() for more detail)
        stepsize_init - dict: {'type': chosen from ['constant', 'harmonic'], 'param': non-negative scalar}
        '''
        
        self.n_steps = n_steps
        # initialize sequence of adjacency matrices
        self.adjacencies = self.topology.collection_duplication(self.topology.adjacencies, n_steps)
        # initialize main and auxiliary states
        self.main3dim = self.states_initialization(state_inits['main'])
        if 'aux' in state_inits:
            self.aux3dim = self.states_initialization(state_inits['aux'])
        # initialize step-size schedule
        self.stepsizes = self.stepsizes_initialization(stepsize_init)

    
    def states_initialization(self, state_init):
        '''Initialize States of all Nodes
        ========== Inputs ==========
        state_init - dict: {'label': init label, 'param': parameter associated to init}
        ========== Outputs ==========
        state3dim - ndarray (n_steps + 1, n_nodes, n_dims): storage of states of all nodes
        ========== Notes ==========
        state_init['label'] is chosen from ['provided', 'minimizer', 'random', 'origin']
            where 'provided': state_init['param'] is ndarray (n_nodes, n_dims)
                  'minimizer': state_init['param'] is ignored and init_states is fetched from functions' minimizer
                  'random': state_init['param'] is the scalar random scale a, i.e., (-a, a)
                  'origin': state_init['param'] is ignored and init_states is zeros
        '''
        
        state_label = state_init['label']
        state_param = state_init['param']
        
        # input preliminary verification
        assert state_label in ['provided', 'minimizer', 'random', 'origin'], "incorrect state init label"
        
        # state initialization
        if state_label == 'provided':
            init_states = np.array(state_param)
        elif state_label == 'minimizer':
            init_states = self.function.minimizers
        elif state_label == 'random':
            assert type(state_param) in [int, float], "incorrect state init type"
            init_states = 2 * state_param * np.random.rand(self.n_nodes, self.n_dims) - state_param
        else:
            init_states = np.zeros((self.n_nodes, self.n_dims))
            
        # check init_states dimension
        assert init_states.shape == (self.n_nodes, self.n_dims), "incorrect init_states dimension"
        
        # state3dim initialization
        state3dim = np.zeros((self.n_steps+1, self.n_nodes, self.n_dims))
        state3dim[0] = init_states  
        return state3dim
        
                    
    def stepsizes_initialization(self, stepsize_init):
        '''Initialize Step-size Schedule
        ========== Inputs ==========
        stepsize_init - dict: {'type': chosen from ['constant', 'harmonic'], 'param': non-negative scalar}
        ========== Outputs ==========
        stepsizes - ndarray (n_steps, ): step-size schedule
        ========== Notes ==========
        if type = 'constant', then stepsizes = [param, param, ...]
        if type = 'harmonic', then stepsizes = param * [1/1, 1/2, 1/3, ...]
        '''
        
        stepsize_type = stepsize_init['type']
        stepsize_param = stepsize_init['param']
        
        # input preliminary verification
        assert stepsize_type in ['constant', 'harmonic'], 'incorrect step-size type'
        assert type(stepsize_param) in [int, float], 'incorrect step-size parameter type'
        assert stepsize_param >= 0.0, 'incorrect step-size parameter value'
        
        # stepsizes initialization
        if stepsize_type == 'constant':
            stepsizes = np.array([stepsize_param] * self.n_steps)
        else:
            stepsizes = stepsize_param * (np.arange(self.n_steps) + 1)
        return stepsizes
    
    
    def metrics_initialization(self):
        '''Initialize Evaluation Metrics'''
        
        self.distances = []
        self.optimality_gaps = []
        self.consensus_diameters = []
        if self.eval_flag and self.function.function_name == 'banknote':
            # initialize accuracies storage
            self.accracies = {'train': [], 'worst_train': [], 'test': [], 'worst_test': []}
            # initialize argmin accuracy storage (how many times each node attain min accuracy)
            dummy_array = [0] * self.n_nodes
            for index in self.adv_indices: 
                dummy_array[index] = None
            self.argmin_accuracy_records = {'train': dummy_array.copy(), 'test': dummy_array.copy()}
            
            
    # ============================== Utility Functions ==============================
        
    def local_observation(self, idx, states, inneighbor_indicators, adv_dict):
        '''Construct Auxiliary or Main States Matrix from 'idx' perspective, 
        i.e., See Vectors transmitted from Adversaries
        ========== Inputs ==========
        idx - nonnegative integer [0, n_nodes): self node index
        states - ndarray (n_nodes, n_dims): original states
        inneighbor_indicators - ndarray (n_nodes, ): in-neighbor indicators 0:not-in-neighbor, 1:in-neighbor
        adv_dict - dict of dicts: adv_dict[send_idx][receive_idx] is ndarray (n_dims, ); adversarial vector
        ========== Outputs ==========
        self_state - ndarray (n_dims, ): self state
        observed_states - ndarray (n_inneighs, n_dims): in-neighbor states (including adversaries)
        '''
        
        full_observed_states = np.copy(states)
        # set self indicator to 'not-in-neighbor'
        indicators = inneighbor_indicators.copy()
        indicators[idx] = 0
        
        # replace some states by adversary states
        for send_idx in adv_dict:
            receive_dict = adv_dict[send_idx]
            if idx in receive_dict:
                full_observed_states[send_idx] = receive_dict[idx]
        
        # get self_state and observed_states
        self_state = full_observed_states[idx]
        inneigh_indices = np.where(indicators == 1)[0]
        observed_states = full_observed_states[inneigh_indices]
        return self_state, observed_states
        
    
    def gradient_step(self, indices, states, stepsize):
        '''Perform Gradient Step Update for Each State
        ========== Inputs ==========
        indices - list of len=n_points: list of node indices
        states - ndarray (n_points, n_dims): state for each node to be evaluated
        stepsize - non-negative scalar: step-size
        ========== Outputs ==========
        updated_states - ndarray (n_points, n_dims): updated states
        '''
        
        # get gradient of each node as ndarray (n_points, n_dims)
        _, _, gradients = self.function.function_eval(indices, states)
        
        # rescale gradients if they are too large
        if self.max_grad_norm is not None:
            grad_norms = np.linalg.norm(gradients, axis=1)
            rescale_indices = np.nonzero(grad_norms > self.max_grad_norm)[0]
            selected_grad_norms = np.expand_dims(grad_norms[rescale_indices], axis=1)
            gradients[rescale_indices] *= (self.max_grad_norm / selected_grad_norms)
        
        # update the state using gradient direction
        updated_states = states - stepsize * gradients
        return updated_states
        
    
    # ============================== Main Framework Functions ==============================
    
    def distributed_algorithm(self, algorithm):
        '''Execute Resilient Distributed Optimization Algorithm
        ========== Inputs ==========
        algorithm - ResilientAlgorithms class: resilient distributed optimization algorithm
        '''

        assert self.n_steps is not None, "initialization() required"
        # initialize evaluation metrics
        self.metrics_initialization()
        
        # start distributed algorithm loop
        for step in range(self.n_steps):
            # display current time-step
            if step % self.show_period == 0: 
                print('---------- Time Step: t = {} ----------'.format(step))
            
            # copy current states, and initialize next states
            main_states = self.main3dim[step].copy()
            next_main_states = np.zeros_like(main_states)
            if algorithm.aux_flag:
                aux_states = self.aux3dim[step].copy()
                next_aux_states = np.zeros_like(aux_states)
                
            # evaluate current states
            if self.eval_flag:
                self.states_evaluation(main_states)
                
            # get adjacency matrix for current time-step
            adjacency = self.adjacencies[step]
                
            # transmit states of adversarial nodes
            main_adv_dict = self.adv.adversary_assignments(main_states, adjacency)
            if algorithm.aux_flag:
                aux_adv_dict = self.adv.adversary_assignments(aux_states, adjacency)

            for reg_node in self.reg_indices:
                # get main and auxiliary states observed by 'reg_node'
                i_main, in_mains = self.local_observation(reg_node, main_states, 
                                                          adjacency[reg_node], main_adv_dict)
                if algorithm.aux_flag:
                    i_aux, in_auxs = self.local_observation(reg_node, aux_states, 
                                                            adjacency[reg_node], aux_adv_dict)
                else:
                    i_aux, in_auxs = None, None

                # implement filters and weighted average (given a resilient algorithm)
                next_i_main, next_i_aux = algorithm.one_step_computation(i_main, in_mains, i_aux, in_auxs)
                
                # store computed states
                next_main_states[reg_node] = next_i_main
                if algorithm.aux_flag:
                    next_aux_states[reg_node] = next_i_aux
                 
            # gradient step (for all regular nodes)
            next_main_states[self.reg_indices] = self.gradient_step(self.reg_indices, 
                             next_main_states[self.reg_indices], self.stepsizes[step])
            
            # save next states
            self.main3dim[step + 1] = next_main_states
            if algorithm.aux_flag:
                self.aux3dim[step + 1] = next_aux_states
                
        # last time-step states evaluation
        if self.eval_flag:
            main_states = self.main3dim[-1].copy()
            self.states_evaluation(main_states)
        
                
    def states_evaluation(self, states):
        '''Evaluate Regular States using distance to minimizer, optimizality gap, and consensus diameter
        ========== Inputs ==========
        states - ndarray (n_nodes, n_dims): state of all nodes
        ========== Outputs ==========
        dist2minimizer - non-negative scalar: \| \bar{x} - x^* \|
        opt_gap - non-negative scalar: f( \bar{x} ) - f^*
        consensus_diam - non-negative scalar: max \| x_i - x_j \|
        accuracies - 4 numbers in %: g_train_accuracy, min_l_train_accuracy, g_test_accuracy, min_l_test_accuracy
        ========== Notes ==========
        \bar{x}: average of states over regular nodes
        g_xxx_accuracy: xxx accuracy calculated using \bar{x}
        min_l_xxx_accuracy: minimum of xxx accuracies calculated at each local state
        '''
        
        # get average state over regular nodes
        reg_states = states[self.reg_indices]
        avg_state = np.average(reg_states, axis=0)
        
        # calculate distance to minimizer: \| \bar{x} - x^* \|
        dist2minimizer = np.linalg.norm(avg_state - self.g_minimizer)
        self.distances.append(dist2minimizer)
        
        # calculate optimality gap: f( \bar{x} ) - f^*
        avg_state_expanded = np.expand_dims(avg_state, axis=0)
        avg_state_repeated = np.repeat(avg_state_expanded, len(self.reg_indices), axis=0)
        _, g_f_vals, _ = self.function.function_eval(self.reg_indices, avg_state_repeated)
        f_val_at_avg_state = g_f_vals[0]
        opt_gap = f_val_at_avg_state - self.g_opt_value
        self.optimality_gaps.append(opt_gap)
        
        # calculate consensus diameter: max \| x_i - x_j \|
        distance_matrix = distance.cdist(reg_states, reg_states, 'euclidean')
        consensus_diam = np.amax(distance_matrix)
        self.consensus_diameters.append(consensus_diam)
        
        # calculate training/test accuracy (if possible)
        if hasattr(self.function, 'models_evaluation'):
            # investigate training accuracy
            l_train_accuracies, g_train_accuracy = \
                self.function.models_evaluation(self.reg_indices, reg_states, 'train')
            min_l_train_accuracy = min(l_train_accuracies)
            idx_min_train = self.reg_indices[np.argmin(l_train_accuracies)]
            
            # investigate test accuracy
            l_test_accuracies, g_test_accuracy = \
                self.function.models_evaluation(self.reg_indices, reg_states, 'test')
            min_l_test_accuracy = min(l_test_accuracies)
            idx_min_test = self.reg_indices[np.argmin(l_test_accuracies)]
            
            # store accuracies
            self.accracies['train'].append(g_train_accuracy)
            self.accracies['worst_train'].append(min_l_train_accuracy)
            self.accracies['test'].append(g_test_accuracy)
            self.accracies['worst_test'].append(min_l_test_accuracy)
            
            # store node attain worst accuracy
            self.argmin_accuracy_records['train'][idx_min_train] += 1
            self.argmin_accuracy_records['test'][idx_min_test] += 1