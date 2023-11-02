### Import General Python Modules
import numpy as np 
import matplotlib.pyplot as plt


class Experiment:
    def __init__(self, alg_framework, algorithms, n_rounds, n_steps, state_inits, stepsize_init):
        '''Running Resilient Distributed Optimization Algorithms Experiments
        ========== Inputs ==========
        alg_framework - DistributedAlgorithmFramework class: object for executing a resilient algorithm
        algorithms - list of ResilientAlgorithms class: resilient algorithms
        n_rounds - positive integer: number of rounds for running each algorithm
        n_steps - positive integer: number of simulation steps
        state_inits - dict of dict: {'main': state_init, 'aux': state_init} where
                                    state_init = {'label': init label, 'param': parameter associated to init}
                                    (see states_initialization() for more detail)
        stepsize_init - dict: {'type': chosen from ['constant', 'harmonic'], 'param': non-negative scalar}
        '''
        
        # classes attributes
        self.alg_framework = alg_framework
        self.algorithms = algorithms
        
        # algorithms configuration
        self.state_inits = state_inits
        self.stepsize_init = stepsize_init
        
        # fundamental attributes
        self.n_nodes = self.alg_framework.n_nodes
        self.n_rounds = n_rounds
        self.n_steps = n_steps
        self.n_algorithms = len(algorithms)
        self.n_xplots = self.n_steps + 1
        
        # network attributes
        self.reg_indices = self.alg_framework.reg_indices
        self.adv_indices = self.alg_framework.adv_indices
        
        # initialize metrics
        self.metrics1 = ['dist_min', 'opt_gap', 'cons_diam']
        self.metrics2 = ['train', 'w_train', 'test', 'w_test']
        self.metric_dict = dict()
        self.worst_counts = None
        
        # benchmark labels
        self.benchmark_labels = {'dist_min': 'min local', 'opt_gap': 'min local', 'cons_diam': None, 
                                 'train': 'benchmark', 'test': 'benchmark'}
        
        # miscellaneous attributes
        self.special_plot_flag = False
        self.k = 10  # top-k worst nodes to be investigated
        
        # run experiments 
        self.run_experiments()
        
    # ============================== Plotting Functions ==============================
        
    def initialize_plots(self):
        '''Initialize Plots'''
        
        metrics = self.metrics1 + self.metrics2 if self.special_plot_flag else self.metrics1
        for metric in metrics:
            self.metric_dict[metric] = dict()
            self.metric_dict[metric]['data'] = np.zeros((self.n_algorithms, self.n_rounds, self.n_steps+1))
            
    
    def initialize_worst_counts(self):
        '''Initialize Worst Accuracy Counts
        Structure of self.worst_counts: [{'train': counts, 'test': counts}, ...] whose len = n_algorithms
        where counts - list of len=n_nodes
        '''
        
        dummy_array = [0 if index in self.reg_indices else None for index in range(self.n_nodes)]
        storage = {'train': dummy_array.copy(), 'test': dummy_array.copy()}
        self.worst_counts = [storage.copy()] * self.n_algorithms
        
        
    def mean_and_error_plot(self, ax, res3dim, worst_flag=False, std_factor=1.0):
        '''Plot mean and error (i.e., std_factor * std) for all algorithms
        ========== Inputs ==========
        ax - plot axis object: figure to be plotted
        res3dim - ndarray (n_algorithms, n_rounds, n_steps): results from all algorithms
        worst_flag - True/False: 
        std_factor - positive scalar: mutiplying factor of standard deviation in plot
        ========== Outputs ==========
        final_values - ndarray (n_algorithms, ): mean value of each algorithm at last time-step
        '''
        
        x_axis = list(range(self.n_xplots))
        linestyle = 'solid'
        
        # mean and error calculation
        mean2dim = np.average(res3dim, axis=1)
        std2dim = np.std(res3dim, axis=1)
        error2dim = std_factor * std2dim
        
        # plot results (mean and error for all algorithms)
        for i in range(self.n_algorithms):
            alg_name = self.algorithms[i].alg_name
            # if plot worst case data
            if worst_flag:
                alg_name += ' (worst)'
                linestyle = 'dotted'
            # plot mean and error for a given algorithm
            ax.plot(x_axis, mean2dim[i], label=alg_name, color='C' + str(i), linestyle=linestyle)
            ax.fill_between(x_axis, mean2dim[i] + error2dim[i], mean2dim[i] - error2dim[i], 
                            alpha=0.30, color='C' + str(i))
        
        # get final values for all algorithms
        final_values = mean2dim[:, -1]
        return final_values
        
    
    def plot_calculation(self, metric, std_factor=1.0):
        '''Plot mean and std of the data for given metric, as well as benchmark value
        ========== Inputs ==========
        metric - string: name of the matric whose the data will be plotted
                (see possible matric names in self.metrics1 and self.metrics2)
        std_factor - positive scalar: mutiplying factor of standard deviation in plot
        ========== Outputs ==========
        final_values - ndarray (n_algorithms, ): mean value of each algorithm at last time-step
        w_final_values - ndarray (n_algorithms, ) or None: mean (worst) value of each algorithm at last time-step
        '''
        
        # initialize plot
        fig, ax = plt.subplots()
        if metric in self.metrics1:
            ax.set_yscale('log')
        
        # mean and error calculation for metric
        res3dim = self.metric_dict[metric]['data']
        final_values = self.mean_and_error_plot(ax, res3dim, False, std_factor)
        w_final_values = None

        # plot additional results ('w_train' or 'w_test')
        if metric in ['train', 'test']:
            extra_metric = 'w_' + metric
            res3dim = self.metric_dict[extra_metric]['data']
            w_final_values = self.mean_and_error_plot(ax, res3dim, True, std_factor)
 
        # plot benchmark value
        bench_value = self.alg_framework.benchmarks[metric]
        if bench_value is not None:
            color_code = 'C' + str(self.n_algorithms)
            ax.axhline(y=bench_value, linestyle='--', label=self.benchmark_labels[metric], color=color_code)
            
        self.metric_dict[metric]['plot'] = (fig, ax)
        return final_values, w_final_values
    
    
    def plot_settings(self, plt_name, axes_limits=None, savefig=False, path_name=''):
        '''Plot Settings and Saving 
        Note: this function will be called outside to set the plot
        ========== Inputs ==========
        plt_name - string: name of plot chosen from ['dist_min', 'opt_gap', 'cons_diam', 
                                                     'train', 'test']
        axes_limits - list: x and y axes limit setting in the form 
                    [xlim_left, xlim_right, ylim_left, ylim_right]
        savefig - True/False: save specified figure
        '''
        
        # retrieve parameters
        function_name = self.alg_framework.function.function_name
        stepsize_type = self.stepsize_init['type']
        alpha = self.stepsize_init['param']
        
        # ========== Plot Settings ==========
        ### Set Title Name
        # set y-label (also appear in title name)
        if plt_name == 'dist_min':
            nickname = 'distance'
            ylabel = 'Distance to the Minimizer'
        elif plt_name == 'opt_gap':
            nickname = 'optimality'
            ylabel = 'Optimality Gap'
        elif plt_name == 'cons_diam':
            nickname = 'diameter'
            ylabel = 'Regular States Diameter'
        elif plt_name == 'train':
            nickname = 'train'
            ylabel = 'Training Accuracy'
        elif plt_name == 'test':
            nickname = 'test'
            ylabel = 'Test Accuracy'
        else:
            NameError('incorrect plot name')
        
        # set step-size name
        if stepsize_type == 'constant':
            alpha_title = r' ($\alpha = $' + str(alpha) + ')'
        elif stepsize_type == 'harmonic':
            alpha_title = r' ($\alpha_k = $' + str(alpha) + '/(k+1)' + ')'
        else:
            alpha_title = ''
            
        # set full title
        title = ylabel + ' at Each Time-step' + alpha_title
        if plt_name in self.metrics2:
            ylabel += ', %'
        
        ### Set x-axis Name
        xlabel = 'Time-step, $k$'
        
        ### Set x-limit and y-limit
        if axes_limits is None:
            axes_limits = [0, self.n_steps, None, None]
        elif axes_limits[0] is None:
            axes_limits[0] = 0
        if plt_name in self.metrics2 and axes_limits[-2:] == [None, None]:
            axes_limits[-2:] = [0, 100]
            
        xlim = axes_limits[:2]
        ylim = axes_limits[-2:]
        
        ### Apply Settings
        fig = self.metric_dict[plt_name]['plot'][0]
        ax = self.metric_dict[plt_name]['plot'][1]
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        ax.grid()
        if plt_name in self.metrics1:
            ax.legend(loc='upper right')
        else:
            ax.legend(loc='lower right')
        
        # ========== Figure Saving ==========
        if savefig:
            print("\n ***** Figure ({}) Saved ***** \n".format(ylabel))
            file_name = function_name + '_' + nickname + '.jpg'
            path = path_name + file_name
            fig.savefig(path, dpi=400)
        
    
    # ============================== Run Experiments ==============================
    
    def update_storages(self, alg_idx, round_idx):
        '''Update data storage using results from experiments; Update self.metric_dict and self.worst_counts
        ========== Inputs ==========
        alg_idx - non-negative integer: algorithm index
        round_idx - non-negative integer: round index
        '''
        
        # insert data into metric_dict
        self.metric_dict['dist_min']['data'][alg_idx, round_idx] = self.alg_framework.distances
        self.metric_dict['opt_gap']['data'][alg_idx, round_idx] = self.alg_framework.optimality_gaps
        self.metric_dict['cons_diam']['data'][alg_idx, round_idx] = self.alg_framework.consensus_diameters
        
        if self.special_plot_flag:
            self.metric_dict['train']['data'][alg_idx, round_idx] = self.alg_framework.accracies['train']
            self.metric_dict['w_train']['data'][alg_idx, round_idx] = self.alg_framework.accracies['worst_train']
            self.metric_dict['test']['data'][alg_idx, round_idx] = self.alg_framework.accracies['test']
            self.metric_dict['w_test']['data'][alg_idx, round_idx] = self.alg_framework.accracies['worst_test']
            
            # update data in self.worst_counts
            counts = self.worst_counts[alg_idx]
            for index in range(self.n_nodes):
                if counts['train'][index] is not None:
                    counts['train'][index] += self.alg_framework.argmin_accuracy_records['train'][index]
                    counts['test'][index] += self.alg_framework.argmin_accuracy_records['test'][index]
        
    
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
    
    
    def run_experiments(self):
        '''Running Experiments'''
        
        metrics = self.metrics1.copy()
        
        # run algorithms with n_rounds for each
        for alg_idx, algorithm in enumerate(self.algorithms):
            # algorithm initialization
            self.alg_framework.initialization(self.n_steps, self.state_inits, self.stepsize_init)
            
            for round_idx in range(self.n_rounds):
                print("\n========== Execution of {}: Round {} ==========".format(algorithm.alg_name, round_idx+1))
                self.alg_framework.distributed_algorithm(algorithm)
    
                # determine if accuracy data exists
                if alg_idx == 0 and round_idx == 0:
                    if self.alg_framework.accracies is not None:
                        self.initialize_worst_counts()
                        self.special_plot_flag = True
                        metrics += ['train', 'test']
                    self.initialize_plots()
                    
                # insert data obtained from self.alg_framework into storages
                self.update_storages(alg_idx, round_idx)
         
        # calculate mean and std over n_rounds, and plot
        print("\n\n========== Show Results ==========")
        print("List of Advesarial Nodes: {}".format(self.adv_indices))
        
        if hasattr(self.alg_framework.function, 'C_params'):
            C = self.alg_framework.function.C_params['optimal']
            print("Regularization Parameter (1/C_optimal): {}".format(1.0 / C))
        
        for metric in metrics:
            print("\n========== Metric: {} ==========".format(metric))
            bench_value = self.alg_framework.benchmarks[metric]
            print("Benchmark Value: {}".format(bench_value))
            final_values, w_final_values = self.plot_calculation(metric)
            print("Mean of Final Values (for each algorithm): {}".format(final_values))
            print("Mean of Worst Final Values (for each algorithm): {}".format(w_final_values))
            
            if metric in ['train', 'test']:
                for alg_index in range(self.n_algorithms):
                    alg_name = self.algorithms[alg_index].alg_name
                    flags = self.get_adv_neighbor_flags(self.worst_counts[alg_index][metric], self.k)
                    print("Worst Node's Adversarial Neighbor Flags for {}: {}".format(alg_name, flags))
            
            
    
 