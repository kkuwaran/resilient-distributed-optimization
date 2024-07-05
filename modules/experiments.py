### Import General Python Modules
from datetime import datetime
import numpy as np 


class Experiment:
    '''Running Resilient Distributed Optimization Algorithms Experiments'''

    def __init__(self, alg_framework, algorithms, sim_init, path_name=None):
        '''Running Resilient Distributed Optimization Algorithms Experiments
        ========== Inputs ==========
        alg_framework - DistributedAlgorithmFramework class: object for executing a resilient algorithm
        algorithms - list of ResilientAlgorithms class: resilient algorithms
        sim_init - dict: dictionary containing initialization parameters for simulation 
                    (inputs to alg_framework.distributed_algorithm() function, and n_rounds)
                    sim_init = {'n_rounds': int, 'n_steps': int, 'state_inits': dict, 'stepsize_init': dict}
                    where state_inits = {'main': state_init, 'aux': state_init}
                          stepsize_init = {'type': str chosen from ['constant', 'harmonic'], 'param': float}
        metrics - list of str: list of metrics to be plotted
        path_name - str | None: path name for saving results
        '''
        
        # classes attributes
        self.alg_framework = alg_framework
        self.algorithms = algorithms
        
        # algorithms configuration
        self.state_inits = sim_init['state_inits']
        self.stepsize_init = sim_init['stepsize_init']
        
        # fundamental attributes
        self.n_nodes = self.alg_framework.n_nodes
        self.n_rounds = sim_init['n_rounds']
        self.n_steps = sim_init['n_steps']
        self.n_algorithms = len(algorithms)
        self.n_xplots = self.n_steps + 1
        
        # network attributes
        self.reg_indices = self.alg_framework.reg_indices
        self.adv_indices = self.alg_framework.adv_indices
        
        # initialize metrics  
        self.metrics1 = self.alg_framework.METRICS1.copy()  # ['dist_min', 'opt_gap', 'cons_diam', 'train_acc', 'train_acc_w', 'test_acc', 'test_acc_w']
        self.metrics2 = self.alg_framework.METRICS2.copy()  # ['argmin_train', 'argmin_test']
        self.metric_dict = None
        self.initialize_metric_dict()
        
        # miscellaneous attributes
        self.k = 10  # top-k worst nodes to be investigated
        
        # run experiments
        self.path_name = path_name
        self.run_experiments()

    # ============================== Initialization Functions ==============================

    def initialize_metric_dict(self):
        '''Initialize Metric Dictionary for storing data and plot
        ========== Notes ==========
        For metrics1,
        - self.metric_dict[metric]['bench'] -> float
        - self.metric_dict[metric]['data'] -> array (n_algorithms, n_rounds, n_steps+1)
        - self.metric_dict[metric]['final'] -> array (n_algorithms, )
        For metrics2,
        - self.metric_dict[metric]['data'] -> array (n_algorithms, n_nodes)
        - self.metric_dict[metric]['flags'] -> list of len=n_algorithms
        '''

        m_dict = dict()

        # initialize b_dict and m_dict for metrics1
        for metric in self.metrics1:
            m_dict[metric] = dict()
            # store initial values
            m_dict[metric]['bench'] = self.alg_framework.benchmarks.get(metric)
            m_dict[metric]['data'] = np.zeros((self.n_algorithms, self.n_rounds, self.n_steps+1))
            m_dict[metric]['final'] = None

        # initialize m_dict for metrics1
        for metric in self.metrics2:
            m_dict[metric] = dict()
            # store initial values
            array = [0 if index in self.reg_indices else None for index in range(self.n_nodes)]
            array = np.repeat([array], self.n_algorithms, axis=0)
            m_dict[metric]['data'] = array
            m_dict[metric]['flags'] = list()
            
        # store results
        self.metric_dict = m_dict

    
    # ============================== Run Experiments ==============================
    
    def update_storages(self, alg_idx, round_idx):
        '''Update data storage using results from experiments; Update self.metric_dict and self.worst_counts
        ========== Inputs ==========
        alg_idx - non-negative integer: algorithm index
        round_idx - non-negative integer: round index
        '''

        for metric in self.metrics1:
            metric_result = self.alg_framework.metric_dict[metric]
            if metric_result:
                self.metric_dict[metric]['data'][alg_idx, round_idx] = metric_result
        
        for metric in self.metrics2:
            # accumulate across round_idx
            counts = self.alg_framework.metric_dict[metric]
            for index in range(self.n_nodes):
                if counts[index] is not None:
                    self.metric_dict[metric]['data'][alg_idx, index] += counts[index]
        
    
    def run_experiments(self):
        '''Running Experiments'''

        for round_idx in range(self.n_rounds):
            print(f"\n========== Round {round_idx+1}/{self.n_rounds} ==========")
            # reinitialize states for each round
            self.alg_framework.initialization(self.n_steps, self.state_inits, self.stepsize_init)
            
            for alg_idx, algorithm in enumerate(self.algorithms):
                self.alg_framework.distributed_algorithm(algorithm)
                # insert data obtained from self.alg_framework into storages
                self.update_storages(alg_idx, round_idx)
        
        print("\n\n========== End of Experiments ==========")
        self.calculate_final_values()
        if self.path_name is not None:
            self.save_results()
            self.write_textfile()


    # ============================== Calculation ==============================

    def calculate_final_values(self):
        '''Calculate Final Values for Metrics1
        ========== Notes ==========
        self.metric_dict[metric]['final'] -> array (n_algorithms, )
        '''

        for metric in self.metrics1:
            # calculate final values
            data3dim = self.metric_dict[metric]['data']
            if not np.array_equal(data3dim, np.zeros_like(data3dim)):
                final_values = np.mean(data3dim[:, :, -1], axis=1)
            else:
                final_values = None
            self.metric_dict[metric]['final'] = final_values


    def get_adv_neighbor_flags(self, counts, k):
        '''Identify whether worst-k indices is a neighbor of an adversary node
        ========== Inputs ==========
        counts - list of len=n_nodes: worst counter for each node
        k - positive integer: number of worst nodes to check
        ========== Outputs ==========
        adv_neighbor_flags - list of len=k: boolean indicating adversary neighbor
        '''
        
        assert len(counts) == self.n_nodes, "incorrect length"
        adv_indices = self.adv_indices
        adjacency = self.alg_framework.adjacencies[0]
        worst_counts = counts.copy()
        
        # convert None in array to '-1'
        for index in adv_indices:
            worst_counts[index] = -1
            
        # get worst 3 regular indices (starting from the worst)
        worst3indices = np.flip(np.argsort(worst_counts)[-k:])
        
        # check if each one is a neighbor of a adversary node
        array_length = worst3indices.size
        adv_neighbor_flags = [False] * array_length
        for i, reg_index in enumerate(worst3indices):
            for adv_index in adv_indices:
                if adjacency[reg_index][adv_index] == 1:
                    adv_neighbor_flags[i] = True
        return adv_neighbor_flags
    

    # ============================== Save Results ==============================

    def write_textfile(self):
        '''Write Text File for Results'''
        
        function_name = self.alg_framework.function.NAME
        file_path = self.path_name + f'{function_name}_results.txt'

        with open(file_path, "w") as file:  
            # adversarial nodes info
            alg_names = [alg.alg_name for alg in self.algorithms]
            file.write(f"List of Advesarial Nodes: {self.adv_indices}\n")
            file.write(f"List of Algorithms: {alg_names}\n")

            # regularization parameter info
            if hasattr(self.alg_framework.function, 'C_params'):
                C_param = self.alg_framework.function.C_params['optimal']
                file.write(f"Regularization Parameter (1/C_optimal): {1.0 / C_param:.4f}\n")

            # write results for each metric
            all_metrics = self.metrics1 + self.metrics2
            for metric in all_metrics:
                file.write(f"\n========== Metric: {metric} ==========\n")
                if metric in self.metrics1:
                    bench_value = self.metric_dict[metric]['bench']
                    final_values = self.metric_dict[metric]['final']
                    file.write(f"Benchmark Value: {bench_value}\n")
                    file.write(f"Mean of Final Values (for each algorithm): {final_values}\n")
                elif metric in self.metrics2:
                    for alg_index, alg_name in enumerate(alg_names):
                        # get worst node's counts and adv neighbor flags
                        counts = self.metric_dict[metric]['data'][alg_index]
                        # Worst Node's Adversarial Neighbor Flags
                        flags = self.get_adv_neighbor_flags(counts, self.k)
                        self.metric_dict[metric]['flags'].append(flags)
                        file.write(f"Adv Neighbor Flags for worst-{self.k} ({alg_name}): {flags}\n")

        print(f"*** results are saved in {file_path} ***")
    

    def save_results(self):
        '''Save Metric Dictionary to a File
        ========== Inputs ==========
        filename - str | None: filename to save results
        '''

        # results preliminary verification
        assert self.metric_dict is not None, "metrics_initialization() required"

        # store results in a dictionary
        results_dict = dict()
        results_dict['eval_timesteps'] = self.alg_framework.eval_timesteps.copy()
        for metric in self.metrics1:
            data3dim = self.metric_dict[metric]['data']
            bench_value = self.metric_dict[metric]['bench']
            if not np.array_equal(data3dim, np.zeros_like(data3dim)):
                results_dict['metric_' + metric] = self.metric_dict[metric]['data']
            if bench_value is not None:
                results_dict['bench_' + metric] = self.metric_dict[metric]['bench']

        # save results to a file
        now = datetime.now()
        dt_string = now.strftime(r"%y%m%d_%H%M%S")
        filename = f'results_{self.alg_framework.function.NAME}_{dt_string}.npz'
        filepath = self.path_name + filename
        np.savez(filepath, **results_dict)
        print(f"*** results are saved in {filepath} ***")