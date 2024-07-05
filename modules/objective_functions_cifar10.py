import time
from datetime import datetime
from tqdm.auto import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras


class DecentralizedCIFAR10:

    NAME = 'cifar10'
    DATASET_NAME = 'cifar10'
    N_CLASS = 10
    N_TRAIN = 50000
    N_VALIDATION = 10000
    INPUT_SHAPE = (32, 32, 3)
    DEFAULT_OPTIMIZER = keras.optimizers.SGD
    DEFAULT_LEARNING_RATE = 1e-3
    DEFAULT_MOMENTUM = 0.9
    BUFFER_SIZE = 1024

    def __init__(self, n_nodes, batch_size=64, bias_flag=False, indices=None, display_flag=False):
        '''CIFAR-10 Dataset Simulation for Resilient Decentralized Optimization
        ========== Inputs ==========
        n_nodes - positive integer: number of nodes in the network
        batch_size - int: number of data points for each pull out
        bias_flag - True/False: distribute training data with same label to each agent
        indices - list of nonnegative numbers in [0, n_nodes): list of regular node indices
        display_flag - True/False: show information
        '''

        # instantiate fundamental attributes
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.bias_flag = bias_flag
        self.indices = indices if indices is not None else list(range(n_nodes))
        self.display_flag = display_flag
        self.function_name = self.DATASET_NAME

        # instantiate datasets, agents and loss function
        datasets = self.dataset_initialization()
        self.ds_train_all = datasets[0]
        self.ds_trains = datasets[1]
        self.ds_val = datasets[2]
        self.agents = self.define_agents()
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy()

        # instantiate number of dimensions (number of trainable parameters)
        self.n_dims = self.count_params()

        # instantiate objects relevant to average
        self.avg_model = self.define_model()
        self.avg_train_acc_metric = keras.metrics.Accuracy()
        self.avg_val_acc_metric = keras.metrics.Accuracy()


    @staticmethod
    def normalize_imgs(ds):
        '''[utility] Normalizes images: `uint8` -> `float32`
        ========== Inputs ==========
        ds - tensorflow dataset: input (image, label) dataset
        ========== Outputs ==========
        ds_out - tensorflow dataset: output (image, label) dataset
        '''

        normalize_img = lambda image, label: (tf.cast(image, tf.float32) / 255., label)
        ds_out = ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        return ds_out


    def dataset_initialization(self):
        '''[main0 internal] Construct CIFAR10 datasets
        ========== Outputs ==========
        ds_train_all - ParallelMapDataset: all training data corresponding to indices
        procesed_ds_trains - list: decentralized datasets, i.e., ds_train for each node
        ds_test - ParallelMapDataset: test data
        '''

        ### initialize training dataset for each node
        chunk_size = self.N_TRAIN // self.n_nodes
        ds_trains = []
        if not self.bias_flag:
            # determine even split
            splits = tfds.even_splits('train', n=self.n_nodes, drop_remainder=True)

            # split dataset to each node
            for node_idx, split in enumerate(splits):
                ds_train = tfds.load(self.DATASET_NAME, split=split, as_supervised=True)
                assert len(ds_train) == chunk_size, "incorrect number of training data points"
                ds_trains.append(ds_train)
        else:
            # get dataset
            ds_train_all = tfds.load(self.DATASET_NAME, split='train', as_supervised=True)
            ds_train_list = list(ds_train_all)
            # sort ds_train_list according to labels
            ds_train_list.sort(key=lambda x: int(x[1]))

            # split ds_train_list to each node
            for node_idx in range(self.n_nodes):
                # determine chunk of ds_train for node_idx
                start = node_idx * chunk_size
                end = (node_idx + 1) * chunk_size if node_idx != (self.n_nodes - 1) else len(ds_train_list)
                ds_train_chunk = ds_train_list[start:end]
                # convert list of data points into tf dataset, and store
                ds_train = tf.data.experimental.from_list(ds_train_chunk)
                ds_trains.append(ds_train)

        # check requirement
        assert len(ds_trains) == self.n_nodes, "incorrect number of training datasets"

        ### process training data, and get ds_train_all
        procesed_ds_trains = []
        count = 0
        for idx, ds_train in enumerate(ds_trains):
            # normalize training data
            ds_train_temp = self.normalize_imgs(ds_train)
            # organize ds_train in batch fashion
            ds_train = ds_train_temp.shuffle(self.BUFFER_SIZE).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            procesed_ds_trains.append(ds_train)

            # accumulate ds_train corresponding to indices
            if idx in self.indices:
                ds_train_all = ds_train_all.concatenate(ds_train_temp) if count != 0 else ds_train_temp
                count += 1

        assert count == len(self.indices), "incorrect number of training datasets"
        assert len(ds_train_all) >= chunk_size * len(self.indices), "incorrect total of data points"
        ds_train_all = ds_train_all.shuffle(self.BUFFER_SIZE).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        ### initialize (one) validation dataset
        ds_test = tfds.load(self.DATASET_NAME, split='test', as_supervised=True)
        ds_test = self.normalize_imgs(ds_test)
        assert len(ds_test) == self.N_VALIDATION, "incorrect validation dataset"
        ds_test = ds_test.shuffle(self.BUFFER_SIZE).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return ds_train_all, procesed_ds_trains, ds_test


    def define_model(self):
        '''[main0 internal] Define CNN model for CIFAR-10 dataset
        ========== Outputs ==========
        model - keras.src.models.sequential.Sequential: CNN model
        '''

        model = keras.Sequential()
        model.add(keras.Input(shape=self.INPUT_SHAPE))
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(self.N_CLASS, activation='softmax'))
        return model


    def define_agents(self):
        '''[main0 internal] Construct agents, one for each node
        ========== Outputs ==========
        agents - list: list of dictionaries with keys
            ['ID', 'model', 'ds_train', 'optimizer', 'train_acc_metric', 'val_acc_metric']
        '''

        # instantiate a list of agents
        agents = list()
        for i in range(self.n_nodes):
            # instantiate an agent
            agents.append({'ID': 'Agent' + str(i+1), 'model': self.define_model(), 'ds_train': self.ds_trains[i],
                          'optimizer': self.DEFAULT_OPTIMIZER(learning_rate=self.DEFAULT_LEARNING_RATE, momentum=self.DEFAULT_MOMENTUM),
                          'train_acc_metric': keras.metrics.Accuracy(),
                          'val_acc_metric': keras.metrics.Accuracy()})
        return agents


    def count_params(self):
        '''Count number of trainable parameters in defined model
        ========== Outputs ==========
        n_param - int: number of trainable parameters
        '''

        # check requirements
        assert isinstance(self.agents, list), "incorrect agents type"

        # get number of parameters for each model
        n_params = list()
        for agent in self.agents:
            model = agent['model']
            n_param = 0
            for layer in model.layers:
                if layer.trainable:
                    n_param += layer.count_params()
            n_params.append(n_param)

        # check resulting number of parameters
        assert len(set(n_params)) == 1, "number of parameters for each model must be equal"
        return n_params[0]


    @staticmethod
    def convert_model_vector(model, vector=None):
        '''[utility] Either put weight vector into model, or get weight vector from model
        ========== Inputs ==========
        model - keras.src.models.functional.Functional: prototype model
        vector - np.ndarray or None: flatten weight
        ========== Notes ==========
        If vector is None, then "get weight vector from model"
        '''

        assert isinstance(model, (keras.src.models.functional.Functional, keras.src.models.sequential.Sequential)), "incorrect model type"
        if vector is None:
            vector = np.concatenate(model.trainable_weights, axis=None)
            return vector
        else:
            assert isinstance(vector, np.ndarray), "incorrect vector type"
            rem_vec = vector.copy()
            weights = list()

            for weight in model.trainable_weights:
                # get desired weight info
                weight_size = int(tf.size(weight))
                weight_shape = weight.shape
                # get weight from vector with correct shape
                cur_vec, rem_vec = np.split(rem_vec, [weight_size])
                reshaped_weight = tf.reshape(cur_vec, shape=weight_shape)
                weights.append(reshaped_weight)
            assert len(rem_vec) == 0, "incorrect vector dimension"

            # replace weights into model
            for layer, weight in enumerate(model.trainable_weights):
                weight.assign(weights[layer])


    def train_baseline(self, n_epoch, optimizer_name, optimizer_config, path_name=None):
        '''Train centralized baseline model
        ========== Inputs ==========
        n_epoch - int: number of training epochs
        optimizer_name - str: optimizer name chosen from ['SGD', 'Adam']
        optimizer_config - dict: configuration for optimizer
        '''

        # check optimizer_name, and initialize optimizer
        if optimizer_name == 'SGD':
            optimizer = keras.optimizers.SGD(**optimizer_config)
        elif optimizer_name == 'Adam':
            optimizer = keras.optimizers.Adam(**optimizer_config)
        else:
            raise NameError("incorrect optimizer name")

        # initilize baseline agent
        agent = {'ID': 'baseline', 'model': self.define_model(), 'ds_train': self.ds_train_all, 'optimizer': optimizer,
                 'train_acc_metric': keras.metrics.Accuracy(), 'val_acc_metric': keras.metrics.Accuracy()}


        def baseline_evaluation(agent, train_accs, test_accs, train_losses, test_losses):
            '''calculate and store training loss, training accuracy and test accuracy'''
            # calculation
            results = self.model_evaluation(agent, modes=['train', 'test'], n_eval_iteration=100)
            # storing
            train_accs.append(results['train'][0])
            test_accs.append(results['test'][0])
            train_losses.append(results['train'][1])
            test_losses.append(results['test'][1])

        # initialize storages
        train_accs = list()
        test_accs = list()
        train_losses = list()
        test_losses = list()

        if self.display_flag:
            print("\n\n===== Start Training Baseline Model =====")

        # evaluate agent before training
        baseline_evaluation(agent, train_accs, test_accs, train_losses, test_losses)

        # start training loop
        for _ in tqdm(range(n_epoch), desc="Epoch"):
            # train agent for one epoch
            self.train_one_epoch(agent)
            # evaluate agent at the end of epoch
            baseline_evaluation(agent, train_accs, test_accs, train_losses, test_losses)

        # store all results
        baseline_dict = {'train_acc': train_accs, 'test_acc': test_accs, 'train_loss': train_losses, 'test_loss': test_losses}

        # save results to a file
        if path_name is not None:
            now = datetime.now()
            dt_string = now.strftime(r"%y%m%d_%H%M%S")
            filename = f'results_{self.NAME}_baseline_{dt_string}.npz'
            filepath = path_name + filename
            np.savez(filepath, **baseline_dict)
            print(f"*** baseline results are saved in {filepath} ***")

        return baseline_dict


    def train_one_epoch(self, agent, stepsize=None):
        '''[main0 external] Training given agent for one epoch, and updating train/test accuracy metrics
        ========== Inputs ==========
        agent - dict: dictionary containing agent information
        stepsize - float: learning rate used for this training epoch
        '''

        # start timer
        if self.display_flag:
            start_time = time.time()

        # extract agent's information
        model = agent['model']
        ds_train = agent['ds_train']
        optimizer = agent['optimizer']

        # define function for one training step
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss_value = self.loss_fn(y, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # modify learning rate
        if stepsize is not None:
            assert isinstance(stepsize, float), "incorrect stepsize type"
            optimizer.learning_rate.assign(stepsize)

        # iterate over the batches of the dataset
        for i, (x_batch_train, y_batch_train) in enumerate(ds_train):
            train_step(x_batch_train, y_batch_train)

        # show training message
        if self.display_flag:
            print(f"Training one epoch for agent with ID: {agent['ID']}", end='; ')
            print(f"Time taken: {time.time() - start_time:.2f}s")


    def models_evaluation(self, indices, states, eval_modes, n_eval_iteration=None):
        '''[main0 external] Evaluation of Logistic Regression Model given Parameters in Network
        ========== Inputs ==========
        indices - list of nonnegative numbers [0, n_nodes): list of node indices
        states - ndarray (n_points, n_dims): state for each node to be evaluated
        modes - list | str: evaluation modes chosen from ['train', 'test'] or both
        n_eval_iteration - int | None: number of evaluation iterations
        ========== Outputs ==========
        results - dict: dictionary containing results for each mode
                {'train': {'local_accuracies': list, 'local_losses': list, 'accuracy_at_avg': float, 'loss_at_avg': float},
                 'test': {'local_accuracies': list, 'local_losses': list, 'accuracy_at_avg': float, 'loss_at_avg': float}}
        '''

        # check inputs requirements
        assert isinstance(indices, list), "incorrect indices type"
        assert isinstance(states, np.ndarray), "incorrect states type"

        assert isinstance(eval_modes, (list, str)), "incorrect modes type"
        if isinstance(eval_modes, str): eval_modes = [eval_modes]
        assert all([mode in ['train', 'test'] for mode in eval_modes]), "incorrect evaluation mode"
        assert n_eval_iteration is None or isinstance(n_eval_iteration, int), "incorrect n_eval_iteration type"

        # initialize results storage
        results = dict()
        for mode in eval_modes:
            results[mode] = {'local_accuracies': list(), 'local_losses': list(), 'accuracy_at_avg': None, 'loss_at_avg': None}

        ### accuracy for each node in 'indices'
        for i, index in enumerate(indices):
            # retrieve information
            agent = self.agents[index]
            state = states[i]

            # calculate and store local accuracy and loss
            self.convert_model_vector(agent['model'], vector=state)
            results_local = self.model_evaluation(agent, eval_modes, n_eval_iteration)

            # store local results
            for mode, (accuracy, loss) in results_local.items():
                results[mode]['local_accuracies'].append(accuracy)
                results[mode]['local_losses'].append(loss)

        ### calculate and store accuracy and loss of average of states
        avg_state = np.average(states, axis=0)  # ndarray (n_dims, )
        self.convert_model_vector(self.avg_model, vector=avg_state)

        agent_temp = {'ID': 'States Average', 'model': self.avg_model,
                      'train_acc_metric': self.avg_train_acc_metric,
                      'val_acc_metric': self.avg_val_acc_metric}
        results_avg = self.model_evaluation(agent_temp, eval_modes, n_eval_iteration)

        # store avg results
        for mode, (accuracy, loss) in results_avg.items():
            results[mode]['accuracy_at_avg'] = accuracy
            results[mode]['loss_at_avg'] = loss

        return results


    def model_evaluation(self, agent, modes, n_eval_iteration=None):
        '''[main1 external] Evaluation of agent's model on training dataset or test dataset
        ========== Inputs ==========
        agent - dict: dictionary containing information specific to given agent
        modes - list | str: evaluation modes chosen from ['train', 'test'] or both
        n_eval_iteration - int | None: number of evaluation iterations
        ========== Outputs ==========
        results - dict: dictionary containing results for each mode
                {'train': (accuracy, loss_value), 'test': (accuracy, loss_value)}
        '''

        # define function for calculating loss_value and predictions (given model)
        @tf.function
        def feedforward(x, y):
            logits = agent['model'](x, training=False)
            loss_value = self.loss_fn(y, logits)
            predictions = tf.math.argmax(logits, axis=1)
            return loss_value, predictions


        # check inputs requirements
        assert isinstance(modes, (list, str)), "incorrect modes type"
        if isinstance(modes, str): modes = [modes]
        assert all([mode in ['train', 'test'] for mode in modes]), "incorrect evaluation mode"
        assert n_eval_iteration is None or isinstance(n_eval_iteration, int), "incorrect n_eval_iteration type"

        # initialize results storage
        results = dict()

        # loop through each mode
        for mode in modes:
            # initialize loss values storage
            loss_values = list()

            # set parameters for each case
            if mode == 'train':
                acc_metric = agent['train_acc_metric']
                dataset = self.ds_train_all
            else:
                acc_metric = agent['val_acc_metric']
                dataset = self.ds_val

            # loop through dataset
            for i, (x_batch, y_batch) in enumerate(dataset):
                if n_eval_iteration is not None and i >= n_eval_iteration:
                    break
                loss_value, predictions = feedforward(x_batch, y_batch)
                acc_metric.update_state(y_batch, predictions)
                loss_values.append(float(loss_value))

            # get accuracy (in decimal) and reset metric state
            accuracy_tf = acc_metric.result()
            acc_metric.reset_state()
            # post-process accuracy to be percentage
            accuracy = float(accuracy_tf) * 100
            # calculate average loss over mini-batches
            avg_loss_value = float(np.average(loss_values))

            # store accuracy and average loss results
            results[mode] = accuracy, avg_loss_value

            # show accuracy result
            if self.display_flag:
                agent_name = agent['ID']
                print(f"Evaluation: {agent_name}; Mode: {mode}", end='; ')
                print(f"Accuracy: {accuracy:.2f}; Loss Value: {avg_loss_value:.4f}")

        return results