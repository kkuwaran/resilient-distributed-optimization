### Import General Modules
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
from numpy import linalg as LA

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from keras.datasets import cifar10
import keras.src
from logistic_regression import Logistic_Regression


class DecentralizedQuadratic:
    def __init__(self, n_nodes, n_dims, quadratic_type, display_flag=False):
        '''Quadratic Functions Simulation for Decentralized Optimization
        ========== Inputs ==========
        n_nodes - positive integer: number of nodes in the network
        n_dims - positive integer: number of dimensions of variables
        quadratic_type - string: type of quadratic functions chosen from ['Q-diag', 'Q-general']
        '''
        
        # fundamental attributes
        self.function_name = 'quadratic'
        self.n_nodes = n_nodes
        self.n_dims = n_dims
        self.quadratic_type = quadratic_type
        # special attributes
        self.display_flag = display_flag
        
        # hyperparameters for construction
        self.scale = 2.5
        self.min_Q_eig = 0.1
        # hyperparameters for checking and displaying
        self.tol_err = 1e-6
        self.n_displays = min(self.n_nodes, 3)

        # quadratic functions construction
        self.Qs = None  # ndarray (n_nodes, n_dims, n_dims)
        self.bs = None  # ndarray (n_nodes, n_dims)
        self.minimizers = None  # ndarray (n_nodes, n_dims)
        self.quadratic_func_gen()
        if self.display_flag:
            self.check_and_display()

        # global optimal attributes
        self.global_minimizer = None  # ndarray (n_dims, )
        self.global_opt_value = None  # scalar
        

    # ============================== Initialization Functions ==============================
    
    def quadratic_func_gen(self):
        '''Construct Quadratic Functions of the form: f(x) = 1/2*x'*Q*x + b'*x
        ========== Outputs ==========
        Qs - ndarray (n_nodes, n_dims, n_dims): matrices representing quadratic coeff.
        bs - ndarray (n_nodes, n_dims): matrices representing linear coeff.
        minimizers - ndarray (n_nodes, n_dims): minimizers of generated quadratic functions
        '''
        
        ### Initialize Variables
        Qs = np.zeros((self.n_nodes, self.n_dims, self.n_dims))
        bs = np.zeros((self.n_nodes, self.n_dims))
        minimizers = np.zeros((self.n_nodes, self.n_dims))

        ### Construction
        for node in range(self.n_nodes):
            if self.quadratic_type == 'Q-diag':
                rand_vec = self.scale * np.random.rand(self.n_dims)
                Qs[node] = np.diag(rand_vec)
            elif self.quadratic_type == 'Q-general':
                # check min eigenvalue > 0.1
                while np.min( LA.eig(Qs[node])[0] ) < self.min_Q_eig:  
                    rand_mat = self.scale * np.random.rand(self.n_dims, self.n_dims)
                    Qs[node] = np.transpose(rand_mat) @ rand_mat  # positive semidefinite
            else: 
                assert False, "invalid type of quadratic functions"
            # random linear term
            bs[node] = -1 * self.scale + 2 * self.scale * np.random.rand(self.n_dims) 
            minimizers[node] = - LA.inv(Qs[node]) @ bs[node]  # minimizer calculation
            
        ### Assignments
        self.Qs = Qs
        self.bs = bs
        self.minimizers = minimizers
            
    
    def check_and_display(self):
        '''Check Generated Quadratic Functions and Display Results'''
        
        ### Validity Check
        grad_norms = np.zeros(self.n_nodes)
        for node in range(self.n_nodes):
            # gradient at minimizer
            gradient = self.Qs[node] @ self.minimizers[node] + self.bs[node]  
            grad_norms[node] = LA.norm(gradient)  # compute gradient norm
        validity = grad_norms < self.tol_err  # check small gradient norm 
        print("\n===== Quadratic Functions Generation =====")
        print("Gradient Norm: \n", grad_norms)
        print("Gradient Validity: \n", validity)
        assert all(validity) is True, "invalid functions construction"
        # check eigenvalues of Qs
        eigen_check = np.zeros((self.n_nodes, self.n_dims))
        for node in range(self.n_nodes): eigen_check[node] = LA.eig(self.Qs[node])[0]
        assert np.all(eigen_check > self.min_Q_eig), "invalid eigenvalues of Qs"
        
        # Display Variables
        print("\n===== Parameters of Local Functions =====")
        print("Q Matrices for first {} nodes: \n{}".format(self.n_displays, self.Qs[:self.n_displays]))
        print("b Vectors for first {} nodes: \n{}".format(self.n_displays, self.bs[:self.n_displays]))
        print("Local Minimizer for first {} nodes: \n{}".format(self.n_displays, self.minimizers[:self.n_displays]))
        
    
    # ============================== Global Optimal Calculation ==============================
    
    def global_optimal_calculation(self, indices=None, attribute_flag=True):
        '''Calculate Global Minimizer and corresponding Optimal Value 
        (w.r.t. Sum of Functions held by 'indices')
        ========== Inputs ==========
        indices - list of nonnegative numbers [0, n_nodes): list of node indices 
        attribute_flag - True/False: set the outputs as attributes
        ========== Outputs ==========
        global_minimizer - ndarray (n_dims, ): global minimizer 
                           i.e., minimizer of sum of functions w.r.t. indices
        global_opt_value - scalar: corresponding optimal value
        '''
        
        if indices is None: indices = list(range(self.n_nodes))
        # calculate global minimizer
        Q_sum = np.sum(self.Qs[indices], axis=0)  # ndarray (n_dims, n_dims)
        b_sum = np.sum(self.bs[indices], axis=0)  # ndarray (n_dims, )
        global_minimizer = -1.0 * LA.inv(Q_sum) @ b_sum 
        # calculate global optimal value
        # Note: f_obj(x) = 1/|n_indices| * (1/2*x'*Q_sum*x + b_sum'*x)
        global_opt_value = 1/len(indices) * (0.5 * global_minimizer.T @ Q_sum @ global_minimizer 
                                             + b_sum @ global_minimizer)

        # optimal model information
        if self.display_flag:
            print("\n===== Global Minimizer and Function Value =====")
            print("Minimizer of Regular Nodes Sum:", global_minimizer)
            print("Optimal Value:", global_opt_value)
        
        # set outputs as attributes or return
        if attribute_flag:
            self.global_minimizer = global_minimizer
            self.global_opt_value = global_opt_value
        else:
            return global_minimizer, global_opt_value
    
        
    # ============================== [Main] Decentralized Functions ==============================
    
    def function_eval(self, indices, states):
        '''Evaluate Local and Global Function Values, and Gradient
        ========== Inputs ==========
        indices - list of nonnegative numbers [0, n_nodes): list of node indices
        states - ndarray (n_points, n_dims): state for each node to be evaluated
        ========== Outputs ==========
        local_func_vals - ndarray (n_points, ): [local] function values evaluated at 'states' 
                          corresponding to 'indices' (different functions for each node)
        global_func_vals - ndarray (n_points, ): [global] function values evaluated at 'states', 
                           i.e., sum of 'indices' nodes functions (same function for each node)
        gradients - ndarray (n_points, n_dims): [local] gradients of corresponding nodes evaluated at 'states'
        '''
        
        assert len(indices) == states.shape[0], "incompatible dimension"
        n_points = len(indices)
        reshaped_states = np.expand_dims(states, axis=2).astype(float)  # column vector (ignore first dim)
        reshaped_states_T = np.expand_dims(states, axis=1).astype(float)  # row vector (ignore first dim)
        reduced_Qs = self.Qs[indices]
        reduced_bs = self.bs[indices]
        
        # [local] function values calculation
        # Note: f(x) = 1/2 * (x_i' * Q_i * x_i) + (b_i' * x_i)
        quadratic_terms = 0.5 * reshaped_states_T @ reduced_Qs @ reshaped_states  # ndarray (n_points, 1, 1)
        linear_terms = reshaped_states_T @ np.expand_dims(reduced_bs, axis=2)  # ndarray (n_points, 1, 1)
        local_func_vals = np.squeeze(quadratic_terms + linear_terms)
        
        # [global] function values calculation
        # Note: f(x) = 1/n_points * (1/2* x'* Q_sum * x + b_sum' * x)
        Q_sum = np.sum(reduced_Qs, axis=0)  # ndarray (n_dims, n_dims)
        b_sum = np.sum(reduced_bs, axis=0)  # ndarray (n_dims, )
        global_func_vals = 1/n_points * (np.diag(0.5 * states @ Q_sum @ states.T) + b_sum @ states.T)
        
        # [local] gradients calculation
        # Note: gradient of quadratic function is (Qx + b)
        gradients = np.squeeze(reduced_Qs @ reshaped_states) + reduced_bs  # ndarray (n_points, n_dims)
        
        return local_func_vals, global_func_vals, gradients
    
    
    
class DecentralizedDataset:
    def __init__(self, function_name, n_nodes, bias_flag=False, n_rand=None, display_flag=False):
        '''BankNotes Dataset Simulation for Resilient Decentralized Optimization
        ### TODO: complete CIFAR-10 part ###
        ========== Inputs ==========
        function_name - string: dataset name chosen from ['banknote', 'cifar10']
        n_nodes - positive integer: number of nodes in the network
        bias_flag - True/False: distribute training data with same label to each agent
        n_rand - nonnegative integer: random state for train_test_split()
        display_flag - True/False: show information
        '''
        
        # fundamental attributes
        self.function_name = function_name
        self.n_nodes = n_nodes
        self.bias_flag = bias_flag
        # special attributes
        self.n_rand = np.random.randint(100, size=1)[0] if None else n_rand
        self.display_flag = display_flag
        
        # important attributes (filled from dataset_initialization())
        self.n_features = None
        self.n_dims = None
        self.X_train_dec = None  # List of ndarrays [(samples, n_features), (samples, n_features), ...]
        self.y_train_dec = None  # List of ndarrays [(samples, 1), (samples, 1), ...]
        self.X_test = None  # ndarray (samples, n_features)
        self.y_test = None  # ndarray (samples, )
        self.C_params = {'optimal': None, 'local': None, 'global': None}  # regularizer
        self.minimizers = None  # minimizer for each agent; ndarray (n_nodes, n_dims)
        self.dataset_initialization()
        
        # global function information (centralized; regular nodes)
        self.model_opt_cen = None
        self.global_minimizer = None  # ndarray (n_dims, )
        self.global_opt_value = None  # scalar
        self.benchmark_accuracies = {'train': None, 'test': None}


    # ============================== Initialization Functions ==============================
    
    def dataset_initialization(self):
        '''Download Dataset and Split into Train, Validation, Test
        ========== Outputs ==========
        X_train_dec - ndarray (n_nodes, samples, n_features): features of training data
        X_val, X_test - ndarray (samples, n_features): features of validation/test data
        y_train_dec - ndarray (n_nodes, samples): labels of training data
        y_val, y_test - ndarray (samples, ): labels of validation/test data
        '''
             
        if self.function_name == 'banknote':
            dataset_name = 'BankNotes'
            n_trains = 1000
            train, test = self.banknotes_initialization(n_trains)
        elif self.function_name == 'cifar10':
            dataset_name = 'CIFAR10'
            train, test = self.cifar10_initialization()
        else:
            raise NameError("dataset not available")
        
        X_train, y_train = train
        # retrieve number of data and number of features
        n_data = X_train.shape[0] + test[0].shape[0]
        n_features = X_train[0].size
        
        if self.display_flag:
            print("\n===== {} Dataset Information =====".format(dataset_name))
            print("Total Number of Data:", n_data)
            print("Number of Features:", n_features)
            print("(Centralized) Training Shape: X-{}; y-{}".format(X_train.shape, y_train.shape))
            print("Test Shape: X-{}; y-{}".format(test[0].shape, test[1].shape))
        
        # sort training data to create bias dataset for each agent
        if self.bias_flag:
            ordering = y_train[:, 0].argsort()
            X_train = X_train[ordering]
            y_train = y_train[ordering]
        
        # distribute training data among nodes
        X_train_dec, y_train_dec = [], []
        n_samples_per_node = int(X_train.shape[0] // self.n_nodes)
        rem_samples = X_train.shape[0] % self.n_nodes
        for node in range(self.n_nodes):
            start_idx = node * n_samples_per_node
            end_idx = (node + 1) * n_samples_per_node
            if node == self.n_nodes - 1: end_idx += rem_samples
            X_train_dec.append(X_train[start_idx:end_idx])
            y_train_dec.append(y_train[start_idx:end_idx])
        
        if self.display_flag:
            n_samples_list = [X_train_i.shape[0] for X_train_i in X_train_dec]
            print("Number of Samples for each Agent: {}".format(n_samples_list))
            print("(Decentralized) Training Shape: X-{}; y-{}".format(X_train_dec[0].shape, y_train_dec[0].shape))
            
        # set attributes
        self.n_features, self.n_dims = n_features, n_features + 1
        self.X_train_dec, self.y_train_dec = X_train_dec, y_train_dec
        self.X_test, self.y_test = test
        self.regularizer_calculation(train)
        if self.bias_flag == False:
            self.local_minimizer_calculation()
            
    
    def banknotes_initialization(self, n_trains):
        '''Download BankNotes Dataset and Partition into Train and Test'''
    
        # download dataset and convert to array
        banknotes = pd.read_csv('../datasets/BankNote_Authentication.csv')
        if self.display_flag:
            sns.pairplot(data=banknotes,hue='class', height=3, aspect=1.5)
            plt.show()
        X = banknotes[['variance', 'skewness', 'curtosis', 'entropy']].to_numpy()
        y = banknotes[['class']].to_numpy()
        trainX, testX, trainy, testy = train_test_split(X, y, train_size=n_trains, random_state=self.n_rand)
        return (trainX, trainy), (testX, testy)
    
    
    def cifar10_initialization(self):
        '''Download CIFAR10 Dataset
        https://ai.plainenglish.io/complete-guide-image-classification-on-cifar-10-chapter-i-multiclass-logistic-regression-525daee0775b
        '''
        
        # load cifar10 dataset
        (trainX, trainy), (testX, testy) = cifar10.load_data()
        # plot first few images
        if self.display_flag:
            for i in range(9):
                # define subplot
                plt.subplot(330 + 1 + i)
                # plot raw pixel data
                plt.imshow(trainX[i])
            # show the figure
            plt.show()
        return (trainX, trainy), (testX, testy)
    
    
    def regularizer_calculation(self, dataset, val_ratio=0.2):
        '''Calculate Regularizers for Logistic Regression Models
        ========== Inputs ==========
        dataset - (X, y): dataset used for validation data splitting
        val_ratio - nonnegative number in [0, 1]: validation data ratio
        ========== Outputs ==========
        C_params - {'optimal': C_opt, 'local': C_local}: regularizers for 
                   centralized non-Byzantine case, and local functions
        ========== Notes ==========
        f_i(w) = alpha/2 w^T w + N * sum_{j=1}^{m_i} log...
        where m_i is no. of data at regular node i-th and N is total no. of nodes
        Thus, f_i(w) = N * Loss_i (1/(N * C_opt)) and nabla f_i(w) = N * nabla Loss(1/(N * C_opt))
        '''
        
        # partition training set into training/validation
        X, y = dataset
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=self.n_rand)
        
        # non-Byzantine centralized training and validation information
        if self.display_flag:
            print("\n===== Non-Byzantine Information =====")
            print("(Non-Byzantine) Training Shape: X-{}; y-{}".format(X_train.shape, y_train.shape))
            print("(Non-Byzantine) Validation Shape: X-{}; y-{}".format(X_val.shape, y_val.shape))
        
        # get scores for all models
        powers = np.arange(-3, 3, 0.5)
        inv_regularizers = 10.0 ** powers
        models = []
        scores = np.zeros_like(powers)
        for i, inv_reg in enumerate(inv_regularizers):
            # Logistic Regression model
            model = LogisticRegression(penalty='l2', C=inv_regularizers[i])
            model.fit(X_train, np.squeeze(y_train))
            models.append(model)
            # calculate score of each LR model
            scores[i] = model.score(X_val, y_val)
            
        # automatically select a good inverse regularizer
        margin = 0.2   # acceptable sacrifice performance from maximum (in %)
        accuracies = scores * 100.0
        threshold = np.max(accuracies) - margin
        for i, accuracy in enumerate(accuracies):
            if accuracy > threshold:
                opt_index = i
                break
        C_opt = inv_regularizers[opt_index]
        C_local = self.n_nodes * C_opt
        
        if self.display_flag:
            regularizers = 1.0 / inv_regularizers
            # plot scores vs regularizers
            print("\n===== Optimal Regularizer Information =====")
            print("Optimal Regularizer: {:.3f}".format(regularizers[opt_index]))
            plt.semilogx(regularizers, scores)
            plt.axhline(y=scores[opt_index], color='r', linestyle='dashed')
            plt.axvline(x=regularizers[opt_index], color='r', linestyle='dashed')
            plt.title("Centralized Model")
            plt.xlabel("Regularizer Values")
            plt.ylabel("Accuracy")
            plt.grid()
            plt.show()
            
            # calculate minimizer and opt_value
            model_opt = models[opt_index]
            minimizer = np.concatenate((model_opt.coef_[0], model_opt.intercept_[0]), axis=None)
            y_transform = np.squeeze(2 * y_train - 1)
            LR_object = Logistic_Regression(X_train, y_transform, 1.0/C_opt)
            opt_value, gradient = LR_object.logistic_loss_and_grad(minimizer)
            # optimal model information
            print("\n===== Optimal Non-Byzantine Model Information =====")
            print("(Non-Byzantine) Optimal Parameters: \n{}".format(minimizer))
            print("(Non-Byzantine) Logistic Loss at Optimal: {:.4f}".format(opt_value) )
            print("(Non-Byzantine) Gradient Norm at Optimal: {:.8f}".format(LA.norm(gradient)) )
            
        # set attributes
        self.C_params['optimal'] = C_opt
        self.C_params['local'] = C_local
        
        
    def local_minimizer_calculation(self):
        '''Calculate Minimizer for Each Agent
        ========== Outputs ==========
        minimizers - ndarray (n_nodes, n_dims): minimizers of local functions
        '''
        
        # calculate minimizers of local models
        minimizers = np.zeros((self.n_nodes, self.n_dims))
        C_local = self.C_params['local']
        for i in range(self.n_nodes):
            # train each local model
            model_local = LogisticRegression(penalty='l2', C=C_local)
            y = np.squeeze(self.y_train_dec[i])
            model_local.fit(self.X_train_dec[i], y)
            # retrieve minimizer from trained local model
            minimizers[i] = np.concatenate((model_local.coef_[0], model_local.intercept_[0]), axis=None)
            
        # show local models information
        if self.display_flag:
            print("\n===== Local Models Information =====")
            alpha_local = 1.0 / C_local
            for i in range(self.n_nodes):
                # Verify Minimizer
                y_transform = np.squeeze(2 * self.y_train_dec[i] - 1)
                LR_object = Logistic_Regression(self.X_train_dec[i], y_transform, alpha_local)
                loss, grad = LR_object.logistic_loss_and_grad(minimizers[i])
                print("Optimal Parameters of {}-th Local Model: {}".format(i+1,  minimizers[i]))
                print("   Logistic Loss of Node {}: {:.4f}".format(i+1, self.n_nodes * loss))
                print("   Gradient Norm of Node {}: {:.8f}".format(i+1, self.n_nodes * LA.norm(grad)))
        
        # set attributes
        self.minimizers = minimizers
    
 
    # ============================== Global Optimal Calculation ==============================
    
    def global_optimal_calculation(self, indices=None, attribute_flag=True):
        '''Calculate Global Minimizer and corresponding Optimal Value 
        (w.r.t. Sum of Functions held by 'indices')
        ========== Inputs ==========
        indices - list of nonnegative numbers [0, n_nodes): list of node indices 
        attribute_flag - True/False: set the outputs as attributes
        ========== Outputs ==========
        global_minimizer - ndarray (n_dims, ): global minimizer 
                           i.e., minimizer of sum of functions w.r.t. indices
        global_opt_value - scalar: corresponding optimal value
        benchmark_accuracies - [benchmark_train_acc, benchmark_test_acc]:
                                benchmark accuracies for train and test datasets calculated using 
                                'global_minimizer' (only if attribute_flag=True)
        ========== Notes ==========
        f(w) = 1/|indices| * sum_{i in indices} f_i(w) 
             = alpha/2 w^T w + N/|indices| * sum_{i in indices} sum_{j=1}^{m_i} log...
             = N/|indices| * Loss_global (|indices|/(N * C_local))
        where Loss_global is the loss of all data in nodes 'indices' and N is total no. of nodes
        '''
        
        if indices is None: indices = list(range(self.n_nodes))
        # aggregate training data
        X_train = np.concatenate([self.X_train_dec[idx] for idx in indices], axis=0)
        y_train = np.concatenate([self.y_train_dec[idx] for idx in indices], axis=0)
        
        # (Byzantine) centralized training and validation information
        if self.display_flag:
            print("\n===== (Byzantine) Centralized Information =====")
            print("Node Indices: {}".format(indices))
            print("Centralized Training Shape: X-{}; y-{}".format(X_train.shape, y_train.shape))

        # calculate centralized optimal model
        C_global = (self.n_nodes / len(indices)) * self.C_params['local']
        model_cen = LogisticRegression(penalty='l2', C=C_global)
        model_cen.fit(X_train, np.squeeze(y_train))
        
        # calculate global_minimizer and global_opt_value
        global_minimizer = np.concatenate((model_cen.coef_[0], model_cen.intercept_[0]), axis=None)
        y_transform = np.squeeze(2 * y_train - 1)
        LR_object = Logistic_Regression(X_train, y_transform, 1.0/C_global)
        global_opt_value, gradient = LR_object.logistic_loss_and_grad(global_minimizer)
        global_opt_value *= (self.n_nodes / len(indices))
        gradient *= (self.n_nodes / len(indices))
        
        # global (Byzantine case) function information
        if self.display_flag:
            print("\n===== Global Minimizer and Function Value =====")
            print("Minimizer of indices Nodes Sum: {}".format(global_minimizer))
            print("Optimal Value: {:.4f}".format(global_opt_value))
            print("Gradient Norm at Optimal: {:.8f}".format(LA.norm(gradient)))
            
        
        # set outputs as attributes or return
        if attribute_flag:
            # calculate benchmark accuracies (centralized model with optimal parameter) for training data and test data
            # note: for training data, use all training data (including data held by adversaries)
            X_data = np.concatenate(self.X_train_dec, axis=0)
            y_data = np.concatenate(self.y_train_dec, axis=0)
            benchmark_train_acc = model_cen.score(X_data, y_data) * 100
            X_data, y_data = self.X_test, self.y_test
            benchmark_test_acc = model_cen.score(X_data, y_data) * 100
            
            # show benchmark accuracies information
            if self.display_flag:
                print("\n===== Evaluation Information (Benchmark) =====")
                print("Benchmark Accuracy (train): {:.2f}".format(benchmark_train_acc))
                print("Benchmark Accuracy (test): {:.2f}".format(benchmark_test_acc))
            
            # set attributes
            self.C_params['global'] = C_global
            self.global_minimizer = global_minimizer
            self.global_opt_value = global_opt_value
            self.benchmark_accuracies['train'] = benchmark_train_acc
            self.benchmark_accuracies['test'] = benchmark_test_acc
        else:
            return global_minimizer, global_opt_value
        

    # ============================== [Main] Decentralized Functions ==============================
    
    def function_eval(self, indices, states):
        '''Evaluate Local and Global Function Values, and Gradient
        ========== Inputs ==========
        indices - list of nonnegative numbers [0, n_nodes): list of node indices
        states - ndarray (n_points, n_dims): state for each node to be evaluated
        ========== Outputs ==========
        local_func_vals - ndarray (n_points, ): [local] function values evaluated at 'states' 
                          corresponding to 'indices' (different functions for each node)
        global_func_vals - ndarray (n_points, ): [global] function values evaluated at 'states', 
                           i.e., sum of 'indices' nodes functions (same function for each node)
        gradients - ndarray (n_points, n_dims): [local] gradients of corresponding nodes evaluated at 'states'
        '''
        
        assert len(indices) == states.shape[0], "incompatible dimension"
        
        # aggregate training data (for global calculation)
        X_train = np.concatenate([self.X_train_dec[idx] for idx in indices], axis=0)
        y_train = np.concatenate([self.y_train_dec[idx] for idx in indices], axis=0)
        
        # determine parameters
        n_points = len(indices)
        alpha_local = 1.0 / self.C_params['local']
        alpha_global = (n_points / self.n_nodes) * alpha_local
        
        # initialize storages
        local_func_vals = np.zeros(n_points)
        global_func_vals = np.zeros(n_points)
        gradients = np.zeros((n_points, self.n_dims))
        
        # "local" functions calculation
        for i, index in enumerate(indices):
            y_transform = np.squeeze(2 * self.y_train_dec[index] - 1)
            LR_object = Logistic_Regression(self.X_train_dec[index], y_transform, alpha_local)
            value, gradient = LR_object.logistic_loss_and_grad(states[i])
            local_func_vals[i] = self.n_nodes * value
            gradients[i] = self.n_nodes * gradient
            
        # "global" function calculation
        y_transform = np.squeeze(2 * y_train - 1)
        LR_object = Logistic_Regression(X_train, y_transform, alpha_global)
        for i in range(n_points):
            value, _ = LR_object.logistic_loss_and_grad(states[i])
            global_func_vals[i] = (self.n_nodes / n_points) * value
        
        return local_func_vals, global_func_vals, gradients
    
        
    def models_evaluation(self, indices, states, eval_mode):
        '''Evaluation of Logistic Regression Model given Parameters in Network
        ========== Inputs ==========
        indices - list of nonnegative numbers [0, n_nodes): list of node indices
        states - ndarray (n_points, n_dims): state for each node to be evaluated
        eval_mode - string: evaluation mode chosen from ['train', 'test']
        ========== Outputs ==========
        local_accuracies - list of len=n_points: local accuracy for each node (use states[i])
        avg_accuracy - scalar: accuracy of states average (use average of states)
        '''
        
        # data and name setting (for each "eval_mode")
        if eval_mode == 'train':
            # note: use all training data (including data held by adversaries)
            X = np.concatenate(self.X_train_dec, axis=0)
            y = np.concatenate(self.y_train_dec, axis=0)
            dataset = (X, y)
        elif eval_mode == 'test':
            dataset = (self.X_test, self.y_test)
        else:
          raise NameError("Invalid Evaluation Mode")
          
        # accuracy for each node in 'indices'
        local_accuracies = []
        for i, index in enumerate(indices):
            names = ['Agent' + str(i), eval_mode]
            accuracy = self.model_evaluation(dataset, states[i], names)
            local_accuracies.append(accuracy)
            
        # accuracy of average of states
        names = ['States Average', eval_mode]
        avg_state = np.average(states, axis=0)  # ndarray (n_dims, )
        avg_accuracy = self.model_evaluation(dataset, avg_state, names)
        
        return local_accuracies, avg_accuracy


    def model_evaluation(self, dataset, state, names=['', '']):
        '''Evaluation of Logistic Regression Model given Parameter
        ========== Inputs ==========
        dataset - (X, y): dataset used to do evaluation
        state - ndarray (n_dims, ): parameter of logistic regression model
        names - [state_name, mode_name]: names for printing result
        '''
        
        # define logistic regression model
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        scoring = get_scorer("accuracy")
        
        # set parameters and calculate accuracy
        X, y = dataset
        model.coef_ = np.expand_dims(state[:-1], axis=0)
        model.intercept_ = np.expand_dims(state[-1], axis=0)
        accuracy = scoring(model, X, y, sample_weight=None) * 100
        
        # show accuracy result
        if self.display_flag:
            state_name, mode_name = names
            print("Accuracy of {} ({}): {:.2f}".format(state_name, mode_name, accuracy))

        return accuracy