import sys
import time
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.insert(0, 'modules')
from objective_functions_cifar10 import DecentralizedCIFAR10


# ===== TEST: constructor, convert_model_vector, train_one_epoch, models_evaluation =====
# set parameters
n_nodes = 10
batch_size = 64
bias_flag = False
display_flag = True
stepsize = 1e-2
n_steps = 10

# set indices
n_index = int(n_nodes // 2)
indices = np.random.choice(n_nodes, size=n_index, replace=False)
indices = np.sort(indices).tolist()
print(f"indices: {indices}")


# ===== test 'constructor' =====
function = DecentralizedCIFAR10(n_nodes, batch_size, bias_flag, indices, display_flag)


# ===== Test: convert_model_vector, train_one_epoch, models_evaluation =====
def get_states(indices):
    states = list()
    for index in indices:
        agent = function.agents[index]
        state = function.convert_model_vector(agent['model'], vector=None)
        states.append(state)
    return np.array(states)

# start timer
start_time_exp = time.time()

# Evaluate old models
states_old = get_states(indices)
print("\nEvaluation (Train/Test) before Training")
_ = function.models_evaluation(indices, states_old, ['train', 'test'])

# Train selected agents
print()
agents = [function.agents[index] for index in indices]
stepsizes = [stepsize] * len(agents)
for _ in range(n_steps):
    for agent, stepsize in zip(agents, stepsizes):
        function.train_one_epoch(agent, stepsize)

# Evaluate new models
states_new = get_states(indices)
print("\nEvaluation (Train/Test) after Training")
_ = function.models_evaluation(indices, states_new, ['train', 'test'])

# stop timer
print(f"\nTime taken: {time.time() - start_time_exp:.2f}s")



# ========================= test 'train_baseline' =========================
# set parameters
n_nodes = 10
n_epoch = 15
batch_size = 64
optimizer_name = 'SGD'
opt_config = {'learning_rate': 0.005, 'momentum': 0.9}
bias_flag = False
display_flag = True
seed = 5


# set random seed and precision
random.seed(seed)
np.random.seed(seed)
tf.experimental.numpy.random.seed(seed)
np.set_printoptions(precision=2)

# set indices
n_index = int(n_nodes // 1.5)
indices = np.random.choice(n_nodes, size=n_index, replace=False)
indices = np.sort(indices).tolist()
print(f"indices: {indices}")

# initialize function
function = DecentralizedCIFAR10(n_nodes, batch_size, bias_flag, indices, display_flag)

# ===== test 'train_baseline' =====
start_time_exp = time.time()
baseline_dict = function.train_baseline(n_epoch, optimizer_name, opt_config)

# show results
print(f"\nTime taken: {time.time() - start_time_exp:.2f}s")
print(f"Training Accuracy: {np.array(baseline_dict['train_acc'])}")
print(f"Test Accuracy: {np.array(baseline_dict['test_acc'])}")
print(f"Training Loss: {np.array(baseline_dict['train_loss'])}")
print(f"Test Loss: {np.array(baseline_dict['test_loss'])}")

# plot results
x = np.arange(n_epoch + 1)
plt.plot(x, baseline_dict['train_acc'], label='train')
plt.plot(x, baseline_dict['test_acc'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, baseline_dict['train_loss'], label='train')
plt.plot(x, baseline_dict['test_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()