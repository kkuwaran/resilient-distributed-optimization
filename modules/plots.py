import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


def compute_ema(arr, s_factor=0.5):
    '''Compute Exponential Moving Average
    ========== Inputs ==========
    arr - list or np.array: list of values
    s_factor - float: smoothing factor (0 < s_factor < 1)
    ========== Outputs ==========
    curr_moving_average - float: current moving average
    '''

    # Check if the list is empty, and initialize the current moving average
    assert len(arr) > 0, "empty list"
    curr_moving_average = arr[0]

    # Loop through the array elements
    for i in range(1, len(arr)):
        # Calculate the exponential average by using the formula
        curr_moving_average = s_factor * arr[i] + (1 - s_factor) * curr_moving_average
    
    return curr_moving_average



class Plottings:
    '''Plottings Class for Plotting Results from Experiments (experiments.py)'''

    X_VALUES_KEY = 'eval_timesteps'

    def __init__(self, filepath, alg_names):
        '''Initialize the object
        ========== Inputs ==========
        filepath - str: path to the npz file containing the results
        alg_names - list: list of algorithm names
        '''

        # fundamental attributes about the file
        self.filepath = filepath
        self.folderpath = '/'.join(filepath.split('/')[:-1]) + '/'

        # fundamental attributes about the algorithms and results
        self.alg_names = alg_names
        self.n_algorithms = len(alg_names)
        self.x_values = None
        self.n_steps = None
        self.n_rounds = None
        
        # dictionary containing the results
        self.metric_dict = dict()
        self.construct_metric_dict(filepath)


    def construct_metric_dict(self, filepath):
        '''Construct Metric Dictionary
        ========== Inputs ==========
        filepath - str: path to the npz file containing the results
        ========== Notes ==========
        The results are stored in the following structure:
        metric_dict = {metric_name: {'data': np.array, 'bench': float}, ...}
        '''
        
        # load the npz file
        npzfile = np.load(filepath, allow_pickle=True)

        for key in npzfile.files:
            # get the x_values
            if key == self.X_VALUES_KEY:
                self.x_values = npzfile[key]
                if self.n_steps is not None:
                    assert self.x_values[-1] == self.n_steps, "incorrect number of steps"
                continue

            # get the metric name
            if key.startswith('metric_'):
                metric_name = key[7:]
            elif key.startswith('bench_'):
                metric_name = key[6:]
            else:
                raise ValueError("incorrect key name")

            # create a dictionary for the metric if it does not exist
            if metric_name not in self.metric_dict:
                self.metric_dict[metric_name] = dict()
            
            # store the results in the dictionary
            if key.startswith('metric_'):
                # get the results for the metric
                res3dim = npzfile[key]
                
                # check the shape of the results
                assert res3dim.shape[0] == len(self.alg_names), "incorrect number of algorithms"
                if self.n_rounds is None:
                    self.n_rounds = res3dim.shape[1]
                    self.n_steps = res3dim.shape[2] - 1
                else:
                    assert self.n_rounds == res3dim.shape[1], "incorrect number of rounds"
                    assert self.n_steps == res3dim.shape[2] - 1, "incorrect number of steps"

                # store the results in the dictionary
                if not np.array_equal(res3dim, np.zeros_like(res3dim)):
                    self.metric_dict[metric_name]['data'] = res3dim.copy()

            elif key.startswith('bench_'):
                # get the benchmark value for the metric
                bench_value = npzfile[key]

                # store the benchmark value in the dictionary
                if bench_value is not None:
                    self.metric_dict[metric_name]['bench'] = bench_value

        # print shape for each metric
        for metric in self.metric_dict:
            data_shape = self.metric_dict[metric]['data'].shape
            bench_value = self.metric_dict[metric].get('bench')
            print(f"metric: {metric}, data shape: {data_shape}, benchmark: {bench_value}")


    # ============================== Plotting Functions ==============================

    def process_results(self, metric_groups, std_factor):
        '''Process Results of Experiments
        ========== Inputs ==========
        metric_groups - list: list of dictionaries containing metrics and plot settings
                    [{'metrics': [metric1, metric2, metric3, ...], 'plt': dict}, ...]
        std_factor - float: mutiplying factor of standard deviation in plot
        '''

        print("\n========== Process Results ==========")
        for group_dict in metric_groups:
            # check if the metric names are correct
            metric_group = group_dict['metrics']
            assert all(metric in self.metric_dict for metric in metric_group), "incorrect metric names"
            
            # calculate mean and std over n_rounds, and plot
            label_exts = group_dict.get('label_exts', [''] * len(metric_group))
            plt_dict = group_dict['plt']
            bench_flag = plt_dict.get('bench_flag', True)
            bench_label = plt_dict.get('bench_label', 'benchmark')
            fig, ax = self.plot_calculation(metric_group, std_factor, label_exts, bench_flag, bench_label)
            # show plots save figures
            self.plot_settings(ax, plt_dict)

            # save figure
            nickname = plt_dict['nickname']
            now = datetime.now()
            dt_string = now.strftime(r"%y%m%d_%H%M%S")
            file_name = f'{nickname}_{dt_string}.jpg'
            path = self.folderpath + file_name
            fig.savefig(path, dpi=400)
            print(f"*** figure are saved in {path} ***")


    def plot_calculation(self, metrics, std_factor=None, label_exts='', 
                         bench_flag=True, bench_label='benchmark'):
        '''Plot mean and std of the data for given metrics, as well as benchmark values
        ========== Inputs ==========
        metrics - list of str: names of the matrics whose the data will be plotted
                (see possible matric names in self.metrics1)
        std_factor - float | None: mutiplying factor of standard deviation in plot
        label_exts - list: list of name extensions for the legend
        ========== Outputs ==========
        self.metric_dict - dict: dictionary containing data, final_values and plot for each metric
        '''

        assert len(metrics) == len(label_exts), "incorrect length"
        linestyles = ['solid', 'dashed', 'dashdot', 'dotted']

        fig, ax = plt.subplots()

        # plot mean and error of result for each metric
        ls_counter = 0  # line-style counter
        for i, metric in enumerate(metrics):
            # calculate and plot mean and error 
            res3dim = self.metric_dict[metric]['data']
            assert res3dim.shape == (self.n_algorithms, self.n_rounds, self.n_steps+1), "incorrect shape"
            linestyle = linestyles[ls_counter % len(linestyles)]
            label_ext = label_exts[i]
            std_factor = std_factor if i == 0 else None  # only plot std for the first metric
            final_values = self.mean_and_error_plot(ax, res3dim, std_factor, linestyle, label_ext)

            # store final_values results and increase counter
            self.metric_dict[metric]['final'] = final_values
            ls_counter += 1

        # plot benchmark value for each metric
        cc_counter = self.n_algorithms  # color-code counter
        for metric in metrics:
            bench_value = self.metric_dict[metric].get('bench')
            if bench_flag and bench_value is not None:
                color_code = 'C' + str(cc_counter)
                ax.axhline(y=bench_value, linestyle='dotted', label=bench_label, color=color_code)
            cc_counter += 1

        return fig, ax 


    def mean_and_error_plot(self, ax, res3dim, std_factor=None, linestyle='solid', label_ext=''):
        '''Plot mean and error (i.e., std_factor * std) for all algorithms
        ========== Inputs ==========
        ax - plot axis object: figure to be plotted
        res3dim - ndarray (n_algorithms, n_rounds, n_steps): results from all algorithms
        std_factor - float | None: mutiplying factor of standard deviation in plot
        linestyle - str: line style for plotting
        label_ext - str: name extension for the legend
        ========== Outputs ==========
        final_values - ndarray (n_algorithms, ): mean value of each algorithm at last time-step
        '''

        # mean and error calculation
        mean2dim = np.average(res3dim, axis=1)
        std2dim = np.std(res3dim, axis=1)
        
        # plot results (mean and error for all algorithms)
        for i in range(self.n_algorithms):
            alg_name = self.alg_names[i]
            mean1dim = mean2dim[i]
            std1dim = std2dim[i]
            color_code = 'C' + str(i)

            # plot mean 
            label_name = alg_name + label_ext
            ax.plot(self.x_values, mean1dim, label=label_name, color=color_code, linestyle=linestyle)
            # plot error (if std_factor is not None)
            if std_factor is not None:
                error1dim = std_factor * std1dim
                ax.fill_between(self.x_values, mean1dim + error1dim, mean1dim - error1dim, 
                                alpha=0.30, color=color_code)
        
        # get final values for all algorithms
        final_values = mean2dim[:, -1]
        return final_values


    # ============================== Manage Figures ==============================

    def plot_settings(self, ax, plt_dict):
        '''Plot Settings and Saving 
        Note: this function will be called from show_result()
        ========== Inputs ==========
        ax - axes object: plot axes
        plt_dict - dict: dictionary containing plot settings
        ========== Notes ==========
        plt_dict.keys() = ['nickname', 'ylog', 'ylabel', 'limit', 'acc_flag', 'legend_loc', 'lr_flag']
        plt_dict['nickname'] - str: nickname for the plot
        plt_dict['ylog'] - True/False: y-axis log scale
        plt_dict['bench_flag'] - True/False: whether to show benchmark value
        plt_dict['ylabel'] - str: y-axis label
        plt_dict['limit'] - list: [xlim_left, xlim_right, ylim_left, ylim_right]
        plt_dict['acc_flag'] - True/False: whether to show accuracy in percentage
        plt_dict['legend_loc'] - str: location of legend, e.g., 'upper right', 'lower right'
        plt_dict['title_ext'] - str: title extension chosen from ['lr', 'atk_param']
        '''
        
        # set full title
        title_ext = plt_dict.get('title_ext', '')
        ylabel = plt_dict['ylabel']
        acc_flag = plt_dict['acc_flag']
        title = ylabel + ' at Each Time-step' + title_ext
        if acc_flag:
            ylabel += ', %'
        
        ### Set x-axis Name
        xlabel = 'Time-step, $k$'
        
        ### Set x-limit and y-limit
        axes_limits = plt_dict['limit']
        if axes_limits is None:
            axes_limits = [0, self.n_steps, None, None]
        elif axes_limits[0] is None:
            axes_limits[0] = 0
        if acc_flag and axes_limits[-2:] == [None, None]:
            axes_limits[-2:] = [0, 100]
            
        xlim = axes_limits[:2]
        ylim = axes_limits[-2:]
        
        ### Apply Settings
        y_log_flag = plt_dict['ylog']
        legend_loc = plt_dict.get('legend_loc', 'upper right')
        if y_log_flag:
            ax.set_yscale('log')
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        ax.legend(loc=legend_loc)
        ax.grid()
        


class Plottings2:
    '''Plottings2 Class for Plotting Results from Experiments (cifar10_experiments.py)'''

    ALG_NAMES = ['DGD', 'R-SDMMFD', 'SDFD', 'CWTM']
    ALG_METRICS = ['metric_train_loss_avg', 'metric_train_loss_w', 'metric_test_loss_avg', 'metric_test_loss_w', 
                   'metric_train_acc_avg', 'metric_train_acc_w', 'metric_test_acc_avg', 'metric_test_acc_w']
    ALG_LINESTYLES = ['solid', 'dashed']

    BASELINE_NAME = 'baseline'
    BASELINE_METRICS = ['train_acc', 'test_acc', 'train_loss', 'test_loss']
    BASELINE_LINESTYLE = 'dotted'
    BASELINE_LABEL = 'benchmark'

    SMOOTHING_FACTOR = 0.2
    

    def __init__(self, directory):
        '''Initialize the object
        ========== Inputs ==========
        directory - str: directory containing the results
        '''

        # fundamental attributes
        self.directory = directory
        self.results_dict = dict()
        self.x_values = None
        self.get_results_from_files(directory)
        self.n_steps = self.x_values[-1]


    def get_results_from_files(self, directory):
        '''Get Results from Files
        ========== Inputs ==========
        identifier - str: identifier for the experiment
        ========== Outputs ==========
        x_values - np.array: array containing the x values for the plots
        results_dict - dict: dictionary containing the results with the following structure:
            results_dict = {metric_name: {alg_name: np.array, ...}, ...}
        '''

        # check the inputs
        assert isinstance(directory, str), 'directory must be a string'
        assert directory.endswith('/'), 'directory must end with /'
        assert os.path.exists(directory), f'{directory} does not exist'
        

        # create a dictionary with the algorithm names as keys and metric dictionaries as values
        for file in os.listdir(directory):
            # skip if the file is not an npz file
            if not file.endswith(".npz"):
                continue

            # load the npz file
            filepath = directory + file
            npzfile = np.load(filepath, allow_pickle=True)
        
            # get the metrics for the algorithm
            alg_name = file.split('_')[2]
            if alg_name in self.ALG_NAMES:
                metrics = self.ALG_METRICS
                
                # get the x_values from the first file
                if self.x_values is None:
                    self.x_values = npzfile['eval_timesteps']
                else:
                    assert np.array_equal(self.x_values, npzfile['eval_timesteps']), 'x_values do not match'

            elif alg_name == self.BASELINE_NAME:
                metrics = self.BASELINE_METRICS
            else:
                raise ValueError(f'incorrect algorithm name: {alg_name}')

            # get the results for each metric
            for metric in metrics:
                # check if the metric is in the file
                assert metric in npzfile, f'{metric} not in {file}'
                self.store_results(npzfile, alg_name, metric)

        # convert the lists to numpy arrays
        self.convert_results_to_arrays()

            
    def store_results(self, npzfile, alg_name, metric):
        '''Store Results in Dictionary
        ========== Inputs ==========
        npzfile - npz file: npz file containing the results
        alg_name - str: name of the algorithm
        metric - str: name of the metric
        ========== Notes ==========
        Store the results in the following structure:
        results_dict = {metric_name: {alg_name: np.array, ...}, ...}
        * If the algorithm is the baseline, store the results in the following structure:
        results_dict = {metric_name_baseline: np.array, ...}
        '''

        if alg_name == 'baseline':
            # add '_baseline' if the algorithm is the baseline
            metric_name = metric + '_baseline'

            # add the results to the dictionary if the metric is not in the dictionary
            if metric_name not in self.results_dict:
                self.results_dict[metric_name] = list()
            self.results_dict[metric_name].append(npzfile[metric])

        else:
            # get the metric name without the 'metric_' prefix
            assert 'metric_' in metric, f'incorrect metric name: {metric}'
            metric_name = metric.split('metric_')[-1]

            # add the results to the dictionary if the metric is not in the dictionary
            if metric_name not in self.results_dict:
                self.results_dict[metric_name] = dict()
            if alg_name not in self.results_dict[metric_name]:
                self.results_dict[metric_name][alg_name] = list()
            self.results_dict[metric_name][alg_name].append(npzfile[metric])


    def convert_results_to_arrays(self):
        '''Convert Results to Arrays'''

        # convert the lists to numpy arrays
        for metric_name, res in self.results_dict.items():
            if self.BASELINE_NAME in metric_name:
                self.results_dict[metric_name] = np.array(res)
                print(f'{metric_name} shape: {self.results_dict[metric_name].shape}')
            else:
                for alg_name, matrix in res.items():
                    self.results_dict[metric_name][alg_name] = np.array(matrix)
                    print(f'{metric_name} {alg_name} shape: {self.results_dict[metric_name][alg_name].shape}')

    
    # ==================== Plot Results ====================

    def plot_results(self, metric_groups):
        '''Plot Results
        ========== Inputs ==========
        metric_groups - list: list of dictionaries containing the metric groups
        ========== Notes ==========
        metric_groups = [{'metrics': [metric1, metric2, ...], 'baseline': baseline_metric, 'plt': plt_dict}, ...]
        plt_dict = {'nickname': str, 'ylog': True/False, 'ylabel': str, 'limit': list, 'acc_flag': True/False, ...}
        '''

        for group_dict in metric_groups:
            # get the metrics and plot settings used in one plot
            metrics = group_dict['metrics']
            baseline_metric = group_dict['baseline']
            plt_dict = group_dict['plt']
            assert len(metrics) <= len(self.ALG_LINESTYLES), 'incorrect number of metrics'

            # initialize the plot
            fig, ax = plt.subplots()

            ### plot the metrics
            for metric_idx, plt_metric in enumerate(metrics):
                # check if the metric is in the results
                assert plt_metric in self.results_dict, f'{plt_metric} not in results_dict'
                algs_dict = self.results_dict[plt_metric]

                # plot the metrics for each algorithm
                for alg_idx, alg_name in enumerate(self.ALG_NAMES):
                    # skip if the algorithm is not in the results
                    if alg_name not in algs_dict:
                        continue

                    # plot the values
                    res_matrix = algs_dict[alg_name]
                    average_values = np.mean(res_matrix, axis=0)
                    label_name = alg_name + plt_dict['label_exts'][metric_idx]
                    color_code = f'C{alg_idx}'
                    linestyle = self.ALG_LINESTYLES[metric_idx]
                    ax.plot(self.x_values, average_values, label=label_name, color=color_code, linestyle=linestyle)
            
            ### add baseline line
            if baseline_metric is not None:
                # check if the baseline metric is in the results
                assert baseline_metric in self.results_dict, f'{baseline_metric} not in results_dict'
                assert 'acc' in baseline_metric or 'loss' in baseline_metric, 'incorrect baseline metric'

                # get the baseline value for the metric
                baseline_curve = np.mean(self.results_dict[baseline_metric], axis=0)
                baseline_value = compute_ema(baseline_curve, self.SMOOTHING_FACTOR)

                # plot the baseline value
                linestyle = self.BASELINE_LINESTYLE
                color_code = f'C{len(self.ALG_NAMES)}'
                label = self.BASELINE_LABEL
                ax.axhline(y=baseline_value, label=label, color=color_code, linestyle=linestyle)      

            ### decorate plot
            self.decorate_plot(ax, plt_dict)

            ### save figure
            file_name = f'{plt_dict['nickname']}.jpg'
            path = self.directory + file_name
            fig.savefig(path, dpi=400)
            print(f"***** {file_name} Saved *****")


    def decorate_plot(self, ax, plt_dict):
        '''Decorate plot
        ========== Inputs ==========
        ax - axes object: plot axes
        plt_dict - dict: dictionary containing plot settings
        ========== Notes ==========
        plt_dict.keys() = ['nickname', 'ylog', 'ylabel', 'limit', 'acc_flag']
        plt_dict['nickname'] - str: nickname for the plot
        plt_dict['ylog'] - True/False: y-axis log scale
        plt_dict['ylabel'] - str: y-axis label
        plt_dict['label_exts'] - list: list of label extensions
        plt_dict['title_ext'] - str: title extension
        plt_dict['limit'] - list: [xlim_left, xlim_right, ylim_left, ylim_right]
        plt_dict['acc_flag'] - True/False: whether to show accuracy in percentage
        '''
        
        ### configure plot settings
        title_extension = plt_dict['title_ext']
        xlabel = 'Epoch'
        ylabel_temp = plt_dict['ylabel']
        acc_flag = plt_dict['acc_flag']
        ylabel = ylabel_temp + ', %' if acc_flag else ylabel_temp
        title = f'{ylabel_temp} at Each Epoch{title_extension}'

        axes_limits = plt_dict['limit'].copy()
        x_right_limit = axes_limits[1]
        if x_right_limit is None or x_right_limit > self.n_steps:
            axes_limits[1] = self.n_steps
        xlim = axes_limits[:2]
        ylim = axes_limits[-2:]
        
        ### apply settings
        if plt_dict['ylog']:
            ax.set_yscale('log')
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim)
        ax.legend()
        ax.grid()