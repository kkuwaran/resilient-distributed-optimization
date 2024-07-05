from datetime import datetime
import numpy as np
from scipy.spatial import distance
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor


class DistributedAlgorithmFramework:
    '''Framework for Distributed Algorithms'''

    MAX_GRAD_NORM = 1e6  # maximum value of gradient norm
    METRICS1 = ['dist_min', 'opt_gap', 'cons_diam', 
                'train_acc', 'train_acc_avg', 'train_acc_w', 
                'test_acc', 'test_acc_avg', 'test_acc_w']
    METRICS2 = ['argmin_train', 'argmin_test']
    NODE_EXECUTION = 'sequential'  # 'parallel' or 'sequential'

    def __init__(self, topology, function, adversaries, eval_info=None):
        '''Initialize Resilient Distributed Optimization Algorithm Attributes
        ========== Inputs ==========
        topology - Topology class: topology of network (i.e., adjacency matrices)
        function - DecentralizedQuadratic or DecentralizedDataset class:
                    function for all nodes in the network
        adversaries - Adversaries class: location and behavior of adversaries
        eval_info - dict | None: dictionary containing information for evaluation
                  {'n_nodes': int | None, 'period': int | None, 'path_name': str | None}
        '''
        
        # relevant objects
        self.topology = topology
        self.function = function
        self.adv = adversaries
        
        # fundamental attributes
        self.n_nodes = self.topology.n_nodes
        self.n_dims = self.function.n_dims
        self.adv_indices = self.adv.adv_indices  # list of adversary indices
        self.reg_indices = self.adv.reg_indices
        
        # global information attributes
        if hasattr(self.function, 'global_optimal_calculation'):
            self.function.global_optimal_calculation(self.reg_indices)
            self.g_minimizer = self.function.global_minimizer  # ndarray (n_dims, )
            self.g_opt_value = self.function.global_opt_value  # scalar
        else:
            self.g_minimizer = None
            self.g_opt_value = None
        
        # algorithm simulation attributes by initialization()
        self.n_steps = None
        self.state_inits = None
        self.stepsize_init = None
        self.adjacencies = None  # ndarray (n_steps, n_nodes, n_nodes)
        self.stepsizes = None  # ndarray (n_steps, )

        # storage of initial states
        self.init_main2dim = None  # ndarray (n_nodes, n_dims)
        self.init_aux2dim = None  # ndarray (n_nodes, n_dims) or None

        # storage of current states
        self.main2dim = None  # ndarray (n_nodes, n_dims)
        self.aux2dim = None  # ndarray (n_nodes, n_dims) or None

        # evaluation attributes by process_eval_info()
        self.eval_flag = False  # boolean: whether evaluation is enabled
        self.eval_indices = None  # list of int: indices of nodes to be evaluated
        self.eval_period = None  # int: evaluation period
        self.eval_timesteps = None  # list of int: time-steps evaluated
        self.process_eval_info(eval_info)
        
        # algorithm evaluation attributes (determined in states_evaluation())
        self.metric_dict = None
        
        # calculate benchmarks for each evaluation metric
        self.benchmarks = {metric: None for metric in self.METRICS1}
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
        
        if self.eval_flag and self.g_minimizer is not None:
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
                
                # store training results
                self.benchmarks['train_acc'] = g_train_acc
                self.benchmarks['train_acc_avg'] = g_train_acc
                self.benchmarks['train_acc_w'] = None

                # store test results
                self.benchmarks['test_acc'] = g_test_acc
                self.benchmarks['test_acc_avg'] = g_test_acc
                self.benchmarks['test_acc_w'] = None
        
        
    # ============================== Initialization Functions ==============================
    
    def initialization(self, n_steps, state_inits, stepsize_init):
        '''[external] Initialize Sequence of Adjacency Matrices, Main/Auxiliary States, and Step-size Schedule
        ========== Inputs ==========
        n_steps - positive integer: number of simulation steps
        state_inits - dict of dict: {'main': state_init, 'aux': state_init} where
                                    state_init = {'label': init label, 'param': parameter associated to init}
                                    (see states_initialization() for more detail)
        stepsize_init - dict: {'type': chosen from ['constant', 'harmonic'], 'param': non-negative scalar}
        '''
        
        ### initialization of algorithm simulation attributes
        self.n_steps = n_steps
        # initialize sequence of adjacency matrices
        self.adjacencies = self.topology.collection_duplication(self.topology.adjacencies, n_steps)
        # initialize main and auxiliary states
        self.state_inits = state_inits
        self.init_main2dim = self.states_initialization(state_inits['main'])
        if 'aux' in state_inits:
            self.init_aux2dim = self.states_initialization(state_inits['aux'])
        # initialize step-size schedule
        self.stepsize_init = stepsize_init
        self.stepsizes = self.stepsizes_initialization(stepsize_init)
        
        ### attack labels if attack_type is 'label_change'
        if self.adv.attack_type == 'label_change':
            assert hasattr(self.function, 'attack_labels'), "attack_labels() required"
            prob = self.adv.attack_param
            self.function.attack_labels(self.adv_indices, prob)
            print("Attack Labels Changed!")

    
    def states_initialization(self, state_init):
        '''Initialize States of all Nodes
        ========== Inputs ==========
        state_init - dict: {'label': init label, 'param': parameter associated to init}
        ========== Outputs ==========
        init_states - ndarray (n_nodes, n_dims): initial states of all nodes
        ========== Notes ==========
        state_init['label'] is chosen from ['provided', 'minimizer', 'random', 'origin']
            where 'provided': state_init['param'] is ndarray (n_nodes, n_dims)
                  'minimizer': state_init['param'] is ignored and init_states is fetched from functions' minimizer
                  'random': state_init['param'] is the scalar random scale a, i.e., (-a, a)
                  'origin': state_init['param'] is ignored and init_states is zeros
        '''
        
        # get state initialization parameters
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
        return init_states.copy()
        
                    
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
        
        # get step-size initialization parameters
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
        '''Initialize Metric Dictionary for Evaluation Metrics'''

        metric_dict = {metric: list() for metric in self.METRICS1}
        dummy_array = [None if idx in self.adv_indices else 0 for idx in range(self.n_nodes)]
        metric_dict.update({metric: dummy_array.copy() for metric in self.METRICS2})
        self.metric_dict = metric_dict.copy()
    
            
    # ============================== Utility Functions ==============================
        
    @staticmethod
    def local_observation(idx, states, inneighbor_indicators, adv_dict):
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
        if self.MAX_GRAD_NORM is not None:
            grad_norms = np.linalg.norm(gradients, axis=1)
            rescale_indices = np.nonzero(grad_norms > self.MAX_GRAD_NORM)[0]
            selected_grad_norms = np.expand_dims(grad_norms[rescale_indices], axis=1)
            gradients[rescale_indices] *= (self.MAX_GRAD_NORM / selected_grad_norms)
        
        # update the state using gradient direction
        updated_states = states - stepsize * gradients
        return updated_states
    

    # # TODO: temporary function!
    # def freeze_state(self, adv_dict, indices=None, state=None):
    #     '''Freeze Adversarial States corresponding to given indices to be 'state'
    #     ========== Inputs ==========
    #     adv_dict - dict of dicts: adv_dict[send_idx][receive_idx] is ndarray (n_dims, ); adversarial vector
    #     indices - list of int: indices of adversarial nodes to be frozen
    #     state - ndarray (n_dims, ): frozen state
    #     ========== Notes ==========
    #     This requires 
    #     > self.freeze_state(main_adv_dict) to be inserted below
    #     main_adv_dict = self.adv.adversary_assignments(main_states, adjacency) in distributed_algorithm() method
    #     > self.freeze_state(aux_adv_dict) to be inserted below
    #     aux_adv_dict = self.adv.adversary_assignments(aux_states, adjacency) in distributed_algorithm() method
    #     '''

    #     # set default indices
    #     if indices is None:
    #         indices = [node_idx for i, node_idx in enumerate(self.adv_indices) if i % 2 == 0]
    #     assert all([idx in self.adv_indices for idx in indices]), "incorrect indices"
        
    #     # set default state
    #     if state is None:
    #         state = np.zeros(self.n_dims)
    #     assert state.shape == (self.n_dims, ), "incorrect state dimension"

    #     # freeze adversarial states
    #     for send_idx in adv_dict:
    #         if send_idx in indices:
    #             for receive_idx in adv_dict[send_idx]:
    #                 adv_dict[send_idx][receive_idx] = state.copy()

    
    # ============================== Main Framework Functions ==============================
    
    def distributed_algorithm(self, algorithm):
        '''Execute Resilient Distributed Optimization Algorithm
        ========== Inputs ==========
        algorithm - ResilientAlgorithms class: resilient distributed optimization algorithm
        '''

        assert self.n_steps is not None, "initialization() required"
        # reset evaluation timesteps and metric dictionary before execution
        self.eval_timesteps = list()
        self.metrics_initialization()
        # reset main and auxiliary states
        self.main2dim = self.init_main2dim.copy()
        if self.init_aux2dim is not None:
            self.aux2dim = self.init_aux2dim.copy()

        # get nodes' indices required update (i.e., follow specified algorithm)
        update_indices = list(range(self.n_nodes))
        if not self.adv.update_adv_flag:
            update_indices = self.reg_indices
        
        # start distributed algorithm loop
        for step in tqdm(range(self.n_steps), desc=f"{algorithm.alg_name} Execution"):
            # copy current states, and initialize next states
            main_states = self.main2dim.copy()
            next_main_states = np.zeros_like(main_states)
            if algorithm.aux_flag:
                aux_states = self.aux2dim.copy()
                next_aux_states = np.zeros_like(aux_states)
                
            # evaluate previous states
            if self.eval_flag and step % self.eval_period == 0:
                self.eval_timesteps.append(step)
                self.states_evaluation(main_states)
                
            # get adjacency matrix for current time-step
            adjacency = self.adjacencies[step]
                
            # transmit states of adversarial nodes
            main_adv_dict = self.adv.adversary_assignments(main_states, adjacency)
            if algorithm.aux_flag:
                aux_adv_dict = self.adv.adversary_assignments(aux_states, adjacency)

            # computation of each node
            def per_node_computation(node_idx):
                # get main and auxiliary states observed by 'reg_node'
                i_main, in_mains = self.local_observation(node_idx, main_states, 
                                                          adjacency[node_idx], main_adv_dict)
                if algorithm.aux_flag:
                    i_aux, in_auxs = self.local_observation(node_idx, aux_states, 
                                                            adjacency[node_idx], aux_adv_dict)
                else:
                    i_aux, in_auxs = None, None

                # implement filters and weighted average (given a resilient algorithm)
                next_i_main, next_i_aux = algorithm.one_step_computation(i_main, in_mains, i_aux, in_auxs)
                
                # store computed states
                next_main_states[node_idx] = next_i_main
                if algorithm.aux_flag:
                    next_aux_states[node_idx] = next_i_aux


            if self.NODE_EXECUTION == 'sequential':
                # execute sequential computation
                for node_idx in update_indices:
                    per_node_computation(node_idx)
            elif self.NODE_EXECUTION == 'parallel':
                # execute parallel computation
                with ThreadPoolExecutor() as executor:
                    executor.map(per_node_computation, update_indices)
            else:
                raise ValueError("incorrect NODE_EXECUTION")


            # gradient step (for all nodes corrsponding to update_indices)
            next_main_states[update_indices] = self.gradient_step(update_indices, 
                             next_main_states[update_indices], self.stepsizes[step])
            
            # save next states
            self.main2dim = next_main_states.copy()
            if algorithm.aux_flag:
                self.aux2dim = next_aux_states.copy()
               
        # last time-step states evaluation
        if self.eval_flag:
            self.eval_timesteps.append(self.n_steps)
            self.states_evaluation(self.main2dim.copy())

        # save results if required
        if self.path_name is not None:
            self.save_results(algorithm.alg_name)
    

    # ============================== Evaluation Functions ==============================

    def process_eval_info(self, eval_info):
        '''Process Evaluation Information
        ========== Inputs ==========
        eval_info - dict | None: dictionary containing information for evaluation
                    {'n_nodes': int | None, 'period': int | None, 'path_name': str | None}
        '''

        if eval_info is not None:
            # input preliminary verification
            assert isinstance(eval_info, dict), "incorrect eval_info type"
            assert all([key in eval_info for key in ['n_nodes', 'period']]), "incorrect keys"

            # process evaluation information
            eval_flag = True
            n_nodes, period = eval_info['n_nodes'], eval_info['period']
            n_eval_nodes = n_nodes if n_nodes is not None else len(self.reg_indices)
            assert n_eval_nodes <= len(self.reg_indices), "incorrect number of evaluation nodes"
            eval_indices = np.random.choice(self.reg_indices, n_eval_nodes, replace=False)
            eval_indices = np.sort(eval_indices).tolist()
            eval_period = period if period is not None else 1
        
            # store evaluation information
            self.eval_flag = eval_flag
            self.eval_indices = eval_indices
            self.eval_period = eval_period
            self.path_name = eval_info.get('path_name', None)

                
    def states_evaluation(self, states):
        '''Evaluate states (corresponding to self.eval_indices) using these metrics:
        distance to minimizer, optimizality gap, and consensus diameter
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
        eval_indices = self.eval_indices.copy()
        eval_states = states[eval_indices]
        avg_state = np.average(eval_states, axis=0)
        
        # calculate distance to minimizer: \| \bar{x} - x^* \|
        if self.g_minimizer is not None:
            dist2minimizer = np.linalg.norm(avg_state - self.g_minimizer)
            self.metric_dict['dist_min'].append(dist2minimizer)
        
        # calculate optimality gap: f( \bar{x} ) - f^*
        if self.g_opt_value is not None:
            avg_state_expanded = np.expand_dims(avg_state, axis=0)
            avg_state_repeated = np.repeat(avg_state_expanded, len(eval_indices), axis=0)
            _, g_f_vals, _ = self.function.function_eval(eval_indices, avg_state_repeated)
            f_val_at_avg_state = g_f_vals[0]
            opt_gap = f_val_at_avg_state - self.g_opt_value
            self.metric_dict['opt_gap'].append(opt_gap)
        
        # calculate consensus diameter: max \| x_i - x_j \|
        distance_matrix = distance.cdist(eval_states, eval_states, 'euclidean')
        consensus_diam = np.amax(distance_matrix)
        self.metric_dict['cons_diam'].append(consensus_diam)
        
        # calculate training/test accuracy (if possible)
        if hasattr(self.function, 'models_evaluation'):
            # investigate training accuracy
            l_train_accuracies, g_train_accuracy = \
                self.function.models_evaluation(eval_indices, eval_states, 'train')
            avg_l_train_accuracy = np.average(l_train_accuracies)
            min_l_train_accuracy = min(l_train_accuracies)
            idx_min_train = eval_indices[np.argmin(l_train_accuracies)]
            
            # investigate test accuracy
            l_test_accuracies, g_test_accuracy = \
                self.function.models_evaluation(eval_indices, eval_states, 'test')
            avg_l_test_accuracy = np.average(l_test_accuracies)
            min_l_test_accuracy = min(l_test_accuracies)
            idx_min_test = eval_indices[np.argmin(l_test_accuracies)]
            
            # store training accuracies
            self.metric_dict['train_acc'].append(g_train_accuracy)
            self.metric_dict['train_acc_avg'].append(avg_l_train_accuracy)
            self.metric_dict['train_acc_w'].append(min_l_train_accuracy)
            # store test accuracies
            self.metric_dict['test_acc'].append(g_test_accuracy)
            self.metric_dict['test_acc_avg'].append(avg_l_test_accuracy)
            self.metric_dict['test_acc_w'].append(min_l_test_accuracy)
            
            # store node attain worst accuracy
            self.metric_dict['argmin_train'][idx_min_train] += 1
            self.metric_dict['argmin_test'][idx_min_test] += 1


    # ============================== Save Results ==============================

    def save_results(self, alg_name):
        '''Save Metric Dictionary to a File
        ========== Inputs ==========
        filename - str | None: filename to save results
        '''
        
        # results preliminary verification
        assert self.metric_dict is not None, "metrics_initialization() required"
        assert self.benchmarks is not None, "benchmarks_calculation() required"

        # get results
        eval_steps = {'eval_timesteps': self.eval_timesteps.copy()}
        metric_results = {'metric_' + key: res for key, res in self.metric_dict.items() if res is not None}
        benchmark_results = {'bench_' + key: res for key, res in self.benchmarks.items() if res is not None}

        # save results to a file
        now = datetime.now()
        dt_string = now.strftime(r"%y%m%d_%H%M%S")
        filename = f'results_{self.function.NAME}_{alg_name}_{dt_string}.npz'
        filepath = self.path_name + filename
        np.savez(filepath, **eval_steps, **metric_results, **benchmark_results)
        print(f"*** results are saved in {filepath} ***")



class DistributedAlgorithmFramework_ML(DistributedAlgorithmFramework):
    '''Class for executing distributed algorithms for ML applications'''

    METRICS1 = DistributedAlgorithmFramework.METRICS1 + ['train_loss', 'train_loss_avg', 'train_loss_w'] + ['test_loss', 'test_loss_avg', 'test_loss_w']

    def __init__(self, topology, function, adversaries, eval_info):
        '''Instantiate attributes from DistributedAlgorithmFramework_ML class
        ========== Inputs ==========
        topology - Topology class: topology of network (i.e., adjacency matrices)
        function - DecentralizedQuadratic or DecentralizedDataset class:
                    function for all nodes in the network
        adversaries - Adversaries class: location and behavior of adversaries
        eval_info - dict: dictionary containing information for evaluation
                  {'n_nodes': int | None, 'period': int | None, 'n_eval_iters': int | None}
        '''

        # instantiate attributes in DistributedAlgorithmFramework class
        super().__init__(topology, function, adversaries, eval_info)

        # instantiate additional attributes
        self.n_eval_iteration = eval_info.get('n_eval_iters')


    def states_initialization(self, state_init):
        '''Initialize states of all nodes extracted from agents' models
        ========== Inputs ==========
        state_init - dict: ignored
        ========== Outputs ==========
        init_states - ndarray (n_nodes, n_dims): initial states of all nodes
        '''

        # check attributes of object self.function
        assert hasattr(self.function, 'convert_model_vector'), "require convert_model_vector method"
        assert hasattr(self.function, 'agents'), "require attribute named agents"

        # set state initialization to be model initialization
        init_states = np.zeros((self.n_nodes, self.n_dims))
        for i, agent in enumerate(self.function.agents):
            init_state = self.function.convert_model_vector(agent['model'], vector=None)
            init_states[i] = init_state

        # check init_states dimension
        assert init_states.shape == (self.n_nodes, self.n_dims), "incorrect init_states dimension"
        print(f"init_states dimension: {init_states.shape}")
        return init_states.copy()


    def gradient_step(self, indices, states, stepsize):
        '''[replace 'gradient_step' method in DistributedAlgorithmFramework class]
        (in this case) Perform one epoch of training for agents associated to given indices
        ========== Inputs ==========
        indices - list of len=n_points: list of node indices
        states - ndarray (n_points, n_dims): state for each node to be evaluated
        stepsize - non-negative scalar: step-size
        ========== Outputs ==========
        updated_states - ndarray (n_points, n_dims): updated states
        '''

        updated_states = np.zeros_like(states)
        for i, node_index in enumerate(indices):
            # get agent and associated state information
            agent = self.function.agents[node_index]
            state = states[i]

            # put state into model
            self.function.convert_model_vector(agent['model'], vector=state)
            # train agent for one epoch
            self.function.train_one_epoch(agent, stepsize)
            # get updated state from model
            updated_state = self.function.convert_model_vector(agent['model'], vector=None)

            # insert updated_state into updated_states
            updated_states[i] = updated_state
        return updated_states


    def states_evaluation(self, states):
        '''Evaluate Regular States using distance to minimizer, optimizality gap, and consensus diameter
        ========== Inputs ==========
        states - ndarray (n_nodes, n_dims): state of all nodes
        ========== Outputs ==========
        avg_train_loss - float: training loss averaged over regular nodes: avg_i f(x_i)
        consensus_diam - non-negative scalar: max \| x_i - x_j \|
        accuracies - 4 numbers in %: g_train_accuracy, min_l_train_accuracy, g_test_accuracy, min_l_test_accuracy
        ========== Notes ==========
        \bar{x}: average of states over regular nodes
        g_xxx_accuracy: xxx accuracy calculated using \bar{x}
        min_l_xxx_accuracy: minimum of xxx accuracies calculated at each local state
        '''

        # check requirements and get states for regular nodes
        assert hasattr(self.function, 'models_evaluation'), "require models_evaluation method"

        # get states for evaluation
        eval_indices = self.eval_indices.copy()
        eval_states = states[eval_indices]

        # calculate consensus diameter: max \| x_i - x_j \|
        distance_matrix = distance.cdist(eval_states, eval_states, 'euclidean')
        consensus_diam = np.amax(distance_matrix)
        self.metric_dict['cons_diam'].append(consensus_diam)

        # Note: models_evaluation output (4 for each mode)
        # {'train': {'local_accuracies': list, 'local_losses': list, 'accuracy_at_avg': float, 'loss_at_avg': float},
        #  'test': {'local_accuracies': list, 'local_losses': list, 'accuracy_at_avg': float, 'loss_at_avg': float}}
        eval_modes = ['train', 'test']
        results = self.function.models_evaluation(eval_indices, eval_states, eval_modes, self.n_eval_iteration)
        for mode in eval_modes:
            mode_results = results[mode]

            # calculate and store accuracy results
            avg_l_accuracy = np.average(mode_results['local_accuracies'])
            min_l_accuracy = np.min(mode_results['local_accuracies'])

            self.metric_dict[mode + '_acc'].append(mode_results['accuracy_at_avg'])
            self.metric_dict[mode + '_acc_avg'].append(avg_l_accuracy)
            self.metric_dict[mode + '_acc_w'].append(min_l_accuracy)

            # calculate and store loss results
            avg_l_loss = np.average(mode_results['local_losses'])
            max_l_loss = np.max(mode_results['local_losses'])

            self.metric_dict[mode + '_loss'].append(mode_results['loss_at_avg'])
            self.metric_dict[mode + '_loss_avg'].append(avg_l_loss)
            self.metric_dict[mode + '_loss_w'].append(max_l_loss)
