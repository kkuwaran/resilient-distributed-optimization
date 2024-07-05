### Import General Python Modules
import random
import numpy as np


class Adversaries:
    def __init__(self, adjacency, F, adv_model, attack_info, display_flag=False):
        '''Initialize Adversaries in a given Network, and Determine Adversary States
        ========== Inputs ==========
        adjacency - ndarray (n_nodes, n_nodes): adjacency matrix with adjacency[i, j] means j -> i
        F - nonnegative integer: parameter for adversary model
        adv_model - string: adversary assumption model chosen from ['total', 'local']
        attack_info - dict: attack information containing keys ['type', 'param']
        display_flag - True/False: show information
        '''

        # check inputs requirements
        assert isinstance(adjacency, np.ndarray), "adjacency must be ndarray"
        assert adjacency.shape[0] == adjacency.shape[1], "adjacency must be square matrix"
        assert isinstance(F, int) and F >= 0, "F must be non-negative integer"
        assert adv_model in ['total', 'local'], "invalid adversary model"
        assert isinstance(attack_info, dict), "attack_info must be dictionary"
        assert all(key in attack_info for key in ['type', 'param']), "incorrect attack_info keys"
        assert isinstance(display_flag, bool), "display_flag must be boolean"

        # check attack_info
        atk_type, param = attack_info['type'], attack_info['param']
        # atk_type - string: attack type chosen from ['random', 'label_change', 'perturbation', 'stubborn']
        if atk_type == 'random':
            # see definition of param in random_attack() method
            assert param is None or param > 0, "invalid byzantine parameter"
            update_adv_flag = False
        elif atk_type == 'label_change':
            # param here is the probability of label change
            assert param is None or (0.0 <= param <= 1.0), "invalid label change parameter"
            update_adv_flag = True
        elif atk_type == 'perturbation':
            # see definition of param in perturbation_attack() method
            assert param is None or param > 0, "invalid perturbation parameter"
            perturbation_mode = attack_info.get('perturbation_mode', 'gaussian')
            broadcast_flag = attack_info.get('broadcast_flag', False)
            assert perturbation_mode in ['fixed-norm', 'gaussian'], "invalid perturbation mode"
            update_adv_flag = True
        elif atk_type == 'stubborn':
            # param here is a tuple of two elements: (based vector, std of Gaussian random vector)
            assert isinstance(param, tuple) and len(param) == 2, "invalid stubborn parameter"
            assert isinstance(param[0], np.ndarray) and isinstance(param[1], float), "invalid stubborn parameter"
            update_adv_flag = False
        else:
            raise ValueError("invalid attack type")
        
        # fundamental attributes
        self.adjacency = adjacency
        self.F = F
        self.adv_model = adv_model
        self.attack_type = atk_type
        self.attack_param = param
        self.n_nodes = self.adjacency.shape[0]

        # auxiliary attributes
        self.random_cache_flag = None
        self.display_flag = display_flag

        # special attributes used in perturbation attack
        if atk_type == 'perturbation':
            self.avg_inneighbors = np.average(np.sum(self.adjacency, axis=1))
            self.perturbation_mode = perturbation_mode
            self.random_cache_flag = broadcast_flag

        # special attributes used in stubborn attack
        if atk_type == 'stubborn':
            based_vector, std = param
            self.stubborn_state = based_vector + std * np.random.randn(*based_vector.shape)
        print(f"Stubborn State: {self.stubborn_state}")
        
        # important attributes
        self.update_adv_flag = update_adv_flag  # flag for updating adversaries (used in DistributedAlgorithmFramework class)
        self.adv_indices = None  # list of adversary indices
        self.adv_indicators = None  # list with 0:non-adversay or 1:adversary
        
        self.adversaries_location()

        # additional attributes
        self.reg_indices = [node for node in range(self.n_nodes) if node not in self.adv_indices]
        
    
    # ============================== Adversaries Placement Functions ==============================
        
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
            raise ValueError("invalid adversary model")
                    
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
            print(f"Advarsary Indices: {adv_indices}")
            print(f"Adversary Indicators: {adv_indicators}")
            print(f"Adversaries Counts: {adversary_counts}")


    # ============================== Byzantine Attack Strategies Functions ==============================
    
    def random_attack(self, states, gamma=None):
        '''(Randomly) Assign an Adversarial State in the (scaled) Box formed by the given states
        adversarial states are selected so that they are unlikely to be filtered out by regular nodes
        ========== Inputs ==========
        states - ndarray (n_points, n_dims): given states
        ========== Output ==========
        adversary_state - ndarray (n_dims, ): assigned adversary state
        '''

        # check attack type
        assert self.attack_type == 'random', "invalid attack type"
        # positive scalar for expansion/contraction factor controlloing range of random values
        gamma = self.attack_param if gamma is None else gamma
        
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
            print(f"Input States: \n{states}")
            print(f"Output Advarsary State: {adversary_state}")
        
        return adversary_state
    

    def perturbation_attack(self, state, gamma=None, cache=None):
        '''Perturb given state by a random vector with fixed-norm or Gaussian distribution
        self.perturbation_mode is chosen from ['fixed-norm', 'gaussian']
        ========== Inputs ==========
        state - ndarray (n_dims, ): given state
        gamma - positive scalar: perturbation factor
        cache - ndarray (n_dims, ) | None: random vector for perturbation
        ========== Output ==========
        adversary_state - ndarray (n_dims, ): perturbed state
        ========== Notes ==========
        if self.perturbation_mode == 'fixed-norm', 
            then gamma is the variance of the random vector normalized by the average in-neighbors
        if self.perturbation_mode == 'gaussian', 
            then gamma is the norm of the random vector normalized by the average in-neighbors and sqrt(n_dims)
        '''

        # check attack type and set (normalized) perturbation size
        assert self.attack_type == 'perturbation', "invalid attack type"
        assert self.perturbation_mode in ['fixed-norm', 'gaussian'], "invalid perturbation mode"
        assert cache is None or isinstance(cache, np.ndarray), "cache must be ndarray"
        n_dims = state.shape[0]
        gamma = self.attack_param if gamma is None else gamma

        # get random vector
        random_vector = np.random.randn(n_dims) if cache is None else cache.copy()
        assert len(random_vector) == n_dims, "random vector must have same length as state"

        # get unit random vector if fixed-norm perturbation
        if self.perturbation_mode == 'fixed-norm':
            random_vector /= np.linalg.norm(random_vector)
            random_vector *= np.sqrt(n_dims)

        # perturb the given state
        perturb_vector = gamma * random_vector * self.avg_inneighbors
        adversary_state = state + perturb_vector

        if self.display_flag:
            print("\n===== Adversarial State Information =====")
            print(f"Input State: {state}")
            print(f"Output Advarsary State: {adversary_state}")

        return adversary_state
    

    # ============================== Adversaries States Assignment Functions ==============================
    
    def adversary_assignments(self, states, adjacency=None):
        '''Value Assignment for each pair of adv_index -> reg_index
        ========== Inputs ==========
        states - ndarray (n_nodes, n_dims): current states of all nodes
        adjacency - ndarray (n_nodes, n_nodes) or None: adjacency matrix (in case topology is time-varying)
        ========== Outputs ==========
        adv_dict - Dict in Dict: adv_dict[idx_send][idx_receive] is ndarray (n_dims, ); adversary state
        '''

        # check inputs requirements
        assert isinstance(states, np.ndarray), "states must be ndarray"
        assert states.shape[0] == self.n_nodes, "states must have n_nodes rows"
        assert states.shape[1] > 0, "states must have at least one column"
        n_dims = states.shape[1]

        # default adjacency matrix
        if adjacency is None:
            adjacency = self.adjacency.copy()
            
        adv_dict = dict()
        for idx_send in self.adv_indices:
            # get out-neighbor indices
            outneighbor_indices = [node for node in range(self.n_nodes) 
                                   if adjacency[node, idx_send] == 1 and node != idx_send]

            # cache random vector
            random_cache = np.random.randn(n_dims) if self.random_cache_flag else None
            
            adv_dict[idx_send] = dict()
            for idx_receive in outneighbor_indices:
                ### get regular in-neighbor indices of "idx_receive" (including itself)
                # get in-neighbor indices (including itself)
                inneighbor_indices = [node for node in range(self.n_nodes) 
                                      if adjacency[idx_receive, node] == 1]
                # get regular in-neighbor indices (including itself)
                reg_inneighbor_indices = [node for node in inneighbor_indices 
                                           if node not in self.adv_indices]
                # adversarial assignment
                reg_inneighbor_states = states[reg_inneighbor_indices]

                # get adversarial state based on attack type
                if self.attack_type == 'random':
                    adv_state = self.random_attack(reg_inneighbor_states, gamma=None)
                elif self.attack_type == 'label_change':
                    adv_state = states[idx_send]
                elif self.attack_type == 'perturbation':
                    adv_state = self.perturbation_attack(states[idx_send], gamma=None, cache=random_cache)
                elif self.attack_type == 'stubborn':
                    adv_state = np.copy(self.stubborn_state)
                adv_dict[idx_send][idx_receive] = adv_state
                
        # display adversarial assignment information
        if self.display_flag: 
            print("\n===== Adversarial Assignment Information =====")
            for idx_send in list(adv_dict.keys()):
                for idx_receive in list(adv_dict[idx_send].keys()):
                    print(f"Adversary Index {idx_send} -> Regular Index {idx_receive}: " +
                          f"State {adv_dict[idx_send][idx_receive]}")            
        return adv_dict
    

    # ============================== Utility Functions ==============================

    def get_out_neighbors_of_adv(self, adv_flag=False, print_flag=False):
        '''get out-neighbors of adversaries
        ========== Inputs ==========
        adv_flag - True/False: get out-neighbors including adversaries or not
        print_flag - True/False: print out-neighbors of adversaries
        ========== Outputs ==========
        adv_out_neighs - Dict: adv_out_neighs[idx_send] is list of indices of out-neighbors of idx_send
        '''

        adv_out_neighs = dict()
        for idx_send in self.adv_indices:
            receive_indices = list()
            for idx_receive in range(self.n_nodes):
                # get indices that will not be included
                indices = [idx_send] if adv_flag else self.adv_indices
                # get indices of idx_send's out-neighbors
                if idx_receive not in indices and self.adjacency[idx_receive, idx_send] == 1:
                    receive_indices.append(idx_receive)
            # store out-neighbors of idx_send
            adv_out_neighs[idx_send] = receive_indices
            # print out-neighbors of idx_send
            if print_flag:
                print(f"Adversary Index {idx_send}: Out-Neighbors {receive_indices}")

        return adv_out_neighs
            