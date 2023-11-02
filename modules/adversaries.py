### Import General Python Modules
import random
import numpy as np


class Adversaries:
    def __init__(self, adjacency, F, adv_model, display_flag=False):
        '''Initialize Adversaries in a given Network, and Determine Adversary States
        ========== Inputs ==========
        adjacency - ndarray (n_nodes, n_nodes): adjacency matrix with adjacency[i, j] means j -> i
        F - nonnegative integer: parameter for adversary model
        adv_model - string: adversary assumption model chosen from ['total', 'local']
        display_flag - True/False: show information
        '''
        
        # fundamental attributes
        self.adjacency = adjacency
        self.F = F
        self.adv_model = adv_model
        self.display_flag = display_flag
        self.n_nodes = self.adjacency.shape[0]
        
        # important attributes
        self.adv_indices = None  # list of adversary indices
        self.adv_indicators = None  # list with 0:non-adversay or 1:adversary
        
        self.adversaries_location()
        
        
    def adversaries_location(self):
        '''Place Adversarial Agents into the Network
        ========== Outputs ==========
        adv_indicators - list of len=n_nodes: adversary indicator where 0:non-adversay, 1:adversary
        '''
        
        if self.adv_model == 'total':
            # choose F nodes from n_nodes nodes
            perm_indices = np.random.permutation(self.n_nodes)
            adv_indices = perm_indices[:self.F]
            
        elif self.adv_model == 'local':
            adv_indices = []
            rem_indices = list(range(self.n_nodes))
            while rem_indices:  # non-empty -> enter loop
                # randomly choose one new adversary from rem_indices
                new_adv_index = random.choice(rem_indices)
                rem_indices.remove(new_adv_index)
                ### try to add idx into adv_indices_temp
                adv_indices_temp = adv_indices.copy()
                adv_indices_temp.append(new_adv_index)
                # count number of adversary in-neighbor
                adv_indicators_temp = [1 if idx in adv_indices_temp else 0 for idx in range(self.n_nodes)]
                local_adv_counts = self.adjacency @ np.array(adv_indicators_temp)
                # set count for adversary indices to 0
                local_adv_counts[adv_indicators_temp] = 0
                # if all counts is less than F then new_adv_index is a valid adversary
                if all(local_adv_counts <= self.F):
                    adv_indices.append(new_adv_index)
        
        else:
            ValueError("invalid adversary model")
                    
        # determine adversary indicators
        adv_indicators = [1 if idx in adv_indices else 0 for idx in range(self.n_nodes)]
        self.adv_indices = adv_indices
        self.adv_indicators = adv_indicators
            
        if self.display_flag:
            ### calculate number of adversaries count for each node
            adversary_counts = [0] * self.n_nodes
            for node in range(self.n_nodes):
                # check if the considering node is an adversary
                if node not in adv_indices:
                    for idx in range(self.n_nodes):
                        # check in-neighbor and adversary
                        if self.adjacency[node, idx] == 1 and idx in adv_indices:
                            adversary_counts[node] += 1
                else:
                    adversary_counts[node] = None
                
            print("\n===== Adversaries Information =====")
            print("Advarsary Indices: {}".format(adv_indices))
            print("Adversary Indicators: {}".format(adv_indicators))
            print("Adversaries Counts: {}".format(adversary_counts))


    # ============================== Adversaries Strategies Functions ==============================
    
    def adversary_assignment_simple(self, states, gamma=1.2):
        '''(Randomly) Assign an Adversarial State in the (scaled) Box formed by the given states
        ========== Inputs ==========
        states - ndarray (n_points, n_dims): given states
        gamma - non-negative scalar: expansion/contraction factor (for controlloing range of random values)
        ========== Output ==========
        adversary_state - ndarray (n_dims, ): assigned adversary state
        '''
        
        # get information; type ndarray (n_dims, )
        min_values = np.amin(states, axis=0)
        max_values = np.amax(states, axis=0)
        middle_values = 0.5 * (min_values + max_values)
        range_values = max_values - min_values
        
        ### assign adversarial value
        n_dims = states.shape[1]
        # construct random values between (-0.5, 0.5) in ndarray (n_dims, )
        random_values = np.random.rand(n_dims) - 0.5
        # assign value in between (min_value - epsilon, max_value + epsilon)
        # where epsilon = 0.5 * (gamma - 1) * range_value
        adversary_state = middle_values + gamma * random_values * range_values
        
        if self.display_flag: 
            print("\n===== Adversarial State Information =====")
            print("Input States: \n{}".format(states))
            print("Output Advarsary State: {}".format(adversary_state))
        
        return adversary_state
    
    
    def adversary_assignments(self, states, adjacency=None):
        '''Value Assignment for each pair of adv_index -> reg_index
        ========== Inputs ==========
        states - ndarray (n_nodes, n_dims): current states of all nodes
        adjacency - ndarray (n_nodes, n_nodes) or None: adjacency matrix (in case topology is time-varying)
        ========== Outputs ==========
        adv_dict - Dict in Dict: adv_dict[idx_send][idx_receive] is ndarray (n_dims, ); adversary state
        '''
        
        if adjacency is None:
            adjacency = self.adjacency.copy()
            
        adv_dict = dict()
        for idx_send in self.adv_indices:
            # get out-neighbor indices
            outneighbor_indices = [node for node in range(self.n_nodes) 
                                   if adjacency[node, idx_send] == 1 and node != idx_send]
            # get regular out-neighbor indices
            reg_outneighbor_indices = [node for node in outneighbor_indices 
                                       if node not in self.adv_indices]
            
            adv_dict[idx_send] = dict()
            for idx_receive in reg_outneighbor_indices:
                ### get regular in-neighbor indices of "idx_receive" (including itself)
                # get in-neighbor indices (including itself)
                inneighbor_indices = [node for node in range(self.n_nodes) 
                                      if adjacency[idx_receive, node] == 1]
                # get regular in-neighbor indices (including itself)
                reg_inneighbor_indices = [node for node in inneighbor_indices 
                                           if node not in self.adv_indices]
                # adversarial assignment
                reg_inneighbor_states = states[reg_inneighbor_indices]
                adv_state = self.adversary_assignment_simple(reg_inneighbor_states)
                adv_dict[idx_send][idx_receive] = adv_state
                
        if self.display_flag: 
            print("\n===== Adversarial Assignment Information =====")
            for idx_send in list(adv_dict.keys()):
                for idx_receive in list(adv_dict[idx_send].keys()):
                    print("Adversary Index {} -> Regular Index {}: \
                          State {}".format(idx_send, idx_receive, adv_dict[idx_send][idx_receive]))              
        return adv_dict