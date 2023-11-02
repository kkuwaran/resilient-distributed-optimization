import math
import numpy as np 
from numpy import linalg as LA

class Topology:
    def __init__(self, topo_type, n_nodes, period, sym_flag=False, **kwargs):
        '''Construct Sequence of Adjacency Matrices
        ========== Inputs ==========
        topo_type - string: topology type chosen from ['complete_ring', 'parital_ring', 'random', 'r-robust']
        n_nodes - positive integer: total number of nodes in network
        period - positive integer: period of topologies (assumed to be of same type)
        sym_flag - True/False: convert directed graph to undirected graph
        kwargs - dict with keys in ['threshold', 'robustness']: see adjacency_matrices_gen()
        ========== Important Attributes ==========
        self.adjacencies - ndarray (period, n_nodes, n_nodes): adjacency matrices in a period
        ========== Notes ==========
        *** A_ij = 1 <-> directed edge from j to i 
        *** period = 1 (time-invariant graph), period > 1 (time-varying graph) 
        *** to generate 'complete_ring' (without switching indices), use 'partial_ring' with period=1
        *** topo_type = 'r-robust' -> automatically generates undirected graph
        '''

        # fundamental attributes
        self.topo_type = topo_type
        self.n_nodes = n_nodes
        self.period = period 
        self.sym_flag = sym_flag

        # error tolerance settings
        self.error_tolerance = 1e-5

        # (asymmetric or symmetric) adjacency matrices construction
        self.adjacencies = self.adjacency_matrices_gen(**kwargs)


    # ============================== Adjacency Matrices Construction Functions ==============================

    def complete_ring(self):
        '''(Adjacency Matrix Construction) Directed Complete Ring Topology
        ========== Outputs ==========
        # adjacency - ndarray (n_nodes, n_nodes): adjacency matrix
        '''

        adjacency = np.identity(self.n_nodes)  # self-loop
        order = np.random.permutation(self.n_nodes)
        for i in range(self.n_nodes):
            # order(1)->order(2), order(2)->order(3), ...
            adjacency[order[(i+1) % self.n_nodes], order[i]] = 1 
        return adjacency
    
    
    def partial_ring(self, p):
        '''(Adjacency Matrix Construction) Time-varying Directed Partial Ring Topology
        ========== Inputs ==========
        self.period - positive integer: period of graph topologies
        p - nonnegative integer [0, period): current index in a period
        ========== Outputs ==========
        adjacency - ndarray (n_nodes, n_nodes): adjacency matrix
        ========== Example ==========
        If n_nodes=5, period=3, we have length=2, last_length=1
        Generate {p=0} 0->1 and 1->2; {p=1} 2->3 and 3->4; {p=2} 4->0
        '''
        
        assert p >= 0 and p < self.period, "invalid index"
        length = math.ceil(self.n_nodes/self.period)  # number of consecutive edges
        adjacency = np.identity(self.n_nodes)  # self-loop
        if p == self.period - 1: length = self.n_nodes - (p * length) # no. of edges for last index
        for i in range(length): adjacency[ (p*length+i+1) % self.n_nodes, p*length+i ] = 1
        return adjacency
    
    
    def random_graph(self, threshold):
        '''(Adjacency Matrix Construction) Directed Random Topology
        ========== Inputs ==========
        threshold - scalar [0, 1]: connection density of random topology
        ========== Outputs ==========
        adjacency - ndarray (n_nodes, n_nodes): adjacency matrix
        '''

        adjacency = np.identity(self.n_nodes)  # self-loop
        adjacency = adjacency + np.random.rand(self.n_nodes, self.n_nodes) # add random numbers
        adjacency = (adjacency > (1 - threshold)).astype(float) # use 'threshold' to determine connection
        return adjacency


    def robust_graph(self, robustness, threshold):
        '''(Adjacency Matrix Construction) Undirected r-robust Graph
        ========== Inputs ==========
        robustness - positive integer: robustness parameter
        threshold - scalar [0, 1]: connection density of added edges
        ========== Outputs ==========
        adjacency - ndarray (n_nodes, n_nodes): adjacency matrix
        '''
        
        min_n_nodes = 2 * robustness - 1  # smallest number of nodes required
        assert self.n_nodes >= min_n_nodes, "not enough nodes"
        adjacency = np.identity(self.n_nodes)  # self-loop
        
        ### Construct F-Elemental Graph (starting from minimal no. of nodes)
        # connected subgraph: use star graph
        f_elemental = np.ones((min_n_nodes, min_n_nodes))
        f_elemental[robustness:, robustness:] = np.identity(robustness-1)
        adjacency[:min_n_nodes, :min_n_nodes] = f_elemental
        # connect new nodes to 'robustness' random available nodes
        for i in range(min_n_nodes, self.n_nodes):
            indices = np.random.permutation(i)[:robustness]
            # add directed edges connecting new node
            adjacency[indices, i] = 1; adjacency[i, indices] = 1; 
        
        ### Add Random Edges according to 'threshold'
        edge_matrix = ( np.triu(np.random.rand(self.n_nodes, self.n_nodes)) 
                       > (1-threshold) ).astype(float)
        edge_matrix_sym = edge_matrix + edge_matrix.T  # symmetric random adjacency matrix
        adjacency = ((adjacency + edge_matrix_sym) > self.error_tolerance).astype(float)
        
        ### Rows & Columns Permutation
        perm_matrix = np.random.permutation(np.identity(self.n_nodes))
        adjacency = perm_matrix @ adjacency @ perm_matrix.T
        return adjacency

    
    def adjacency_matrices_gen(self, **kwargs):
        '''(Main) Adjacency Matrix Construction
        ========== Inputs (kwargs) ==========
        threshold - scalar [0, 1]: connection density for ['random', 'r-robust']
        robustness - positive integer: robustness parameter for ['r-robust']
        ========== Outputs ==========
        adjacencies - ndarray (period, n_nodes, n_nodes): adjacency matrices in a period
        '''
        
        # initialize adjacency matrix collection
        adjacencies = np.zeros((self.period, self.n_nodes, self.n_nodes))
        
        if self.topo_type == 'complete_ring':  # complete ring topology
            for p in range(self.period): adjacencies[p] = self.complete_ring();
            if self.sym_flag: adjacencies = self.symmetric_transform(adjacencies)
            
        elif self.topo_type == 'partial_ring':  # partial ring topology 
            for p in range(self.period): adjacencies[p] = self.partial_ring(p)
            if self.sym_flag: adjacencies = self.symmetric_transform(adjacencies)

        elif self.topo_type == 'random':  # random [period] topologies
            threshold = kwargs.get('threshold')
            assert threshold is not None, "hyperparameters required"
            ### Ensure Periodically Connected Property
            adjacency_product = np.identity(self.n_nodes)  # create product of adjacency matrices
            connected = False  # used to break while loop
            while not connected:
                for p in range(self.period):
                    adjacency = self.random_graph(threshold)
                    # create symmetric matrix
                    if self.sym_flag: 
                        adjacency = ((adjacency + np.transpose(adjacency)) > 0.5).astype(float)
                    adjacencies[p] = adjacency
                    # accumulate product
                    adjacency_product = adjacency @ adjacency_product  # @ is a shortcut for matmul
                # check all elements of product^n is nonzero 
                if (np.count_nonzero( LA.matrix_power(adjacency_product, self.n_nodes) ) 
                    == self.n_nodes * self.n_nodes):
                    connected = True  # connected property is True
                else:  # start new construction again
                    adjacency_product = np.identity(self.n_nodes)
                    
        elif self.topo_type == 'r-robust':  # r-robust graphs
            threshold = kwargs.get('threshold'); robustness = kwargs.get('robustness');
            assert threshold is not None and robustness is not None, "hyperparameters required"
            for p in range(self.period): adjacencies[p] = self.robust_graph(robustness, threshold);
            # Note: adjacency matrices are automatically symmetric
            
        else:  # invalid topology type
            raise NameError("invalid topology type")
            
        return adjacencies
            

    # ============================== Helping Functions ==============================
    
    def symmetric_transform(self, adjacencies):
        '''Transform Asymmetric Adjacency Collection into Symmetric Adjacency Collection
        ========== Inputs ==========
        adjacencies - ndarray (n_nodes, n_nodes) or ndarray (period, n_nodes, n_nodes): 
                      adjacency matrix collection
        ========== Outputs ==========
        sym_adjacencies - ndarray (n_nodes, n_nodes) or ndarray (period, n_nodes, n_nodes): 
                          symmetric adjacency matrix collection
        '''
        
        if adjacencies.ndim == 2:
            adjacencies_T = np.transpose(adjacencies)
        elif adjacencies.ndim == 3:
            adjacencies_T = np.transpose(adjacencies, (0, 2, 1))
        else:
          raise ValueError("invalid number of dimensions")
        sym_adjacencies = (adjacencies + adjacencies_T > 0.5).astype(float)
        return sym_adjacencies
    
    
    def collection_duplication(self, matrices, time_horizon):
        '''Duplicate Matrix Collection throughout Time Horizon
        ========== Inputs ==========
        matrices - ndarray (n_nodes, n_nodes) or ndarray (period, n_nodes, n_nodes): matrix collection
        time_horizon - positive integer: total number of time-steps
        ========== Outputs ==========
        duplicated_matrices - ndarray (time_horizon, n_nodes, n_nodes): matrices for all time-steps
        '''
        
        n_nodes = matrices.shape[-1]
        # initialize storage
        duplicated_matrices = np.zeros((time_horizon, n_nodes, n_nodes))
        # calculation
        if matrices.ndim == 2:
            for t in range(time_horizon): 
                duplicated_matrices[t] = matrices
        elif matrices.ndim == 3:
            period = matrices.shape[0]
            for t in range(time_horizon):
                period_idx = t % period
                duplicated_matrices[t] = matrices[period_idx]
        else:
            raise ValueError("invalid number of dimensions")
        return duplicated_matrices
    

    # ============================== Stochastic Matrix Construction Functions ==============================

    def stochastic_matrices_gen(self, adjacency, stoch_type):
        '''Construct Column/Doubly Stochastic Matrix from given Adjacency Matrix
        Note: Column Stochastic generated by Random; Doubly Stochastic generated by Metropolis Hastings
        ========== Inputs ==========
        adjacency - ndarray (n_nodes, n_nodes): adjacency matrix
        stoch_type - string: stochastic matrix type chosen from ['column', 'doubly']
        ========== Outputs ==========
        stochastic_mat - ndarray (n_nodes, n_nodes): stochastic matrix
        '''
        
        n_nodes = adjacency.shape[0]
        ### Column Stochastic Matrices Construction [Random]
        if stoch_type == 'column':
            # construct unnormalized stochastic matrix
            stochastic_mat = adjacency * np.random.rand(n_nodes, n_nodes) 
            # construct normalized stochastic matrix
            column_sum = np.sum(stochastic_mat, axis=0, keepdims=True)
            stochastic_mat = stochastic_mat / column_sum
        ### Doubly Stochastic Matrices Construction [Metropolis Hastings]
        elif stoch_type == 'doubly':
            # check symmetric adjacency matrix
            sym_error_mat = adjacency - np.transpose(adjacency)
            assert LA.norm(sym_error_mat) < self.error_tolerance, "required symmetric adjacency matrix"
            ### Construction Part
            # calculate max degree
            mask = adjacency * (1 - np.identity(n_nodes))
            degree = np.sum(mask, axis=1, keepdims=True)
            max_degree = np.maximum(degree, np.transpose(degree))
            # construct doubly stochastic matrix
            weights = 1 / (1 + max_degree) * mask
            diag_vec = 1 - np.sum(weights, axis=1)
            stochastic_mat = weights + np.diag(diag_vec)
        else:
          raise NameError("invalid stochastic matrix type")
        return stochastic_mat


    # ============================== Stochastic Matrix Properties Checking Functions ==============================

    def properties_check(self, stochastic_mat, stoch_type):
        '''Check Stochastic Matrix Properties
        ========== Inputs ==========
        stochastic_mat - ndarray (n_nodes, n_nodes): stochastic matrix
        stoch_type - string: stochastic matrix type chosen from ['column', 'doubly']
        '''
        
        n_nodes = stochastic_mat.shape[0]
        # check if stochastic_mat is column stochastic
        cs_error_vec = np.sum(stochastic_mat, axis=0) - np.ones(n_nodes)
        result = LA.norm(cs_error_vec) < self.error_tolerance
        if stoch_type == 'doubly':
            # check if stochastic_mat is row stochastic
            rs_error_vec = np.sum(stochastic_mat, axis=1) - np.ones(n_nodes)
            rs_result = LA.norm(rs_error_vec) < self.error_tolerance
            # check if stochastic_mat is symmetric
            sym_error_mat = stochastic_mat - np.transpose(stochastic_mat)
            sym_result = LA.norm(sym_error_mat) < self.error_tolerance
            result = result and rs_result and sym_result
        assert result, "invalid stochastic matrix"


    # ============================== Stochastic Matrix Parameters ==============================

    def parameters(self, stochastic_mat):
        '''Calculate Spectral Gap [1-lambda_2(W)] and Identity Gap [2-Norm of I-W] of Stochastic Matrix
        Note: If "Stochastic_mat" has 3 dims, calculate the parameters of the product of W
        ========== Inputs ==========
        stochastic_mat - ndarray (n_nodes, n_nodes) or ndarray (period, n_nodes, n_nodes): stochastic matrices
        ========== Outputs ==========
        delta - non-negative scalar: Spectral Gap
        beta - non-negative scalar: Identity Gap
        '''
        
        n_nodes = stochastic_mat.shape[-1]
        if stochastic_mat.ndim == 2:
            # check doubly stochastic property
            self.properties_check(stochastic_mat, 'doubly')
            product_mat = stochastic_mat
        elif stochastic_mat.ndim == 3:
            # check doubly stochastic property and calculate product
            product_mat = np.identity(n_nodes)
            for p in range(stochastic_mat.shape[0]):
                self.properties_check(stochastic_mat[p], 'doubly')
                product_mat = stochastic_mat[p] @ product_mat
        else:
            raise ValueError("invalid number of dimensions")

        # calculate spectral gap
        eig, _ = LA.eig(product_mat)
        sorted_eig = np.sort( np.abs(eig) )
        delta = 1 - sorted_eig[-2]
        # calculate identity gap
        beta = LA.norm(np.identity(n_nodes) - product_mat, ord=2)
        return delta, beta