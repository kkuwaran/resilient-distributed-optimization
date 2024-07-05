# resilient-distributed-optimization
### Resilient Distributed Optimization Algorithms/Algorithmic Framework 
In this repository, we focus on addressing Byzantine-resilient distributed optimization problems. Our work revolves around the implementation of the **RE**silient **D**istributed **GR**adient-descent **A**lgorithmic **F**ramework **(REDGRAF)** [1], which encompasses various resilient algorithms. These algorithms are designed to overcome Byzantine failures, ensuring robust distributed optimization. 

Implemented Algorithms:
-  **Simultaneous Distance-MixMax Filtering Dynamics (SDMMFD)** [2, 3]
-  **Reduced Simultaneous Distance-MixMax Filtering Dynamics (R-SDMMFD)** 
-  **Simultaneous Distance Filtering Dynamics (SDFD)** [2]
-  **Coordinate-wise Trimmed Mean (CWTM)** [4-9]
-  **Coordinate-wise Median (CWMed)** [4]
-  **Resilient Vector Optimization (RVO)** [10, 11]

### Experimentation and Analysis
We conduct comprehensive experiments to evaluate the performance of each algorithm. Our experiments include three key components:
- **Quadratic Functions Experiment:** We evaluate how these algorithms perform in optimizing distributed quadratic functions, providing insights into their convergence and efficiency in a controlled environment.
- **Banknote Authentication Experiment (Binary Classification Task):** We apply resilient algorithms to a real-world low-dimensional classification problem—authenticating banknotes. This experiment assesses their effectiveness in tackling small-scale optimization challenges.
- **CIFAR-10 Classification Experiment (Multi-Class Classification Task):** We address a high-dimensional classification problem by classifying images from the CIFAR-10 dataset. This experiment evaluates the algorithms' effectiveness in handling large-scale optimization challenges.

### Dependency Plots
To gain insights into the behavior of these algorithms, we provide plots that showcase their dependency on the constant step-size. These plots specifically illustrate the following key metrics:
- **Convergence Rate:** How quickly the algorithms converge to a region containing the optimal solution.
- **Convergence Radius:** The reach of the algorithms in terms of convergence.
- **Approximate Consensus Diameter:** The extent to which the states of regular agents close together.
- **Training Accuracy:** The classification performance on training data.
- **Test Accuracy:** The classification performance on test data.

### Folder Structure
- Folder **modules**
  - **main_script.ipynb:** Configures and runs the quadratic functions and banknote authentication experiments as shown in [1].
  - **cifar10_experiments.ipynb / cifar10_experiments.py:** Configures and runs CIFAR-10 experiments.
  - **plot.py:** Plots results from experiments.
  - **experiments.py:** Executes resilient algorithms and saves results.
  - **algorithmic_framework.py:** Implements REDGRAF.
  - **resilient_algorithms.py:** Implements resilient algorithms.
  - **adversaries.py:** Initializes Byzantine agents and determines their behavior.
  - **objective_functions.py:** Initializes (distributed) quadratic functions and the banknote authentication task.
  - **objective_functions_cifar10.py:** Initializes (distributed) CIFAR-10 classification task.
  - **topology_generation.py:** Initializes network topology.
  - **dependency_plots.py:** Plots variable dependencies considered in the theoretical analysis in [1].
  - Folder **utilities:** Contains external subroutine functions.
- Folder **figures**
  - **!exp_quadratic:** Stores main results and figures from quadratic functions experiments.
  - **!exp_banknote:** Stores main results and figures from banknote authentication experiments.
  - **!exp_cifar10:** Stores main results and figures from CIFAR-10 experiments.
  - **variables_dependency:** Stores figures generated from dependency_plots.py.
- Folder **datasets**
  - **BankNote_Authentication.csv:** Stores the dataset used for the banknote authentication task.
- Folder **testers**
  - **algorithmic_framework_test.py:** Script for testing algorithmic_framework.py.
  - **resilient_algorithms_test.py:** Script for testing resilient_algorithms.py.
  - **adversaries_test.py:** Script for testing adversaries.py.
  - **quadratic_functions_test.py:** Script for testing the DecentralizedQuadratic class in objective_functions.py.
  - **banknote_functions_test.py:** Script for testing the DecentralizedBankNotes class in objective_functions.py.
  - **cifar10_functions_test.py:** Script for testing the DecentralizedCIFAR10 class in objective_functions_cifar10.py.
  - **topology_generation_test.py:** Script for testing topology_generation.py.
  - **logistic_regression_test.py:** Script for testing logistic_regression.py (in the utilities folder).


### References
#### Our Relevant Works
[1] K. Kuwaranancharoen and S. Sundaram, *On the Geometric Convergence of Byzantine-Resilient Distributed Optimization Algorithms*, 2023, [Paper Link](https://arxiv.org/abs/2305.10810). <br>
[2] K. Kuwaranancharoen, L. Xin, and S. Sundaram, *Scalable Distributed Optimization of Multi-Dimensional Functions Despite Byzantine Adversaries*, in IEEE Transactions on Signal and Information Processing over Networks, 2024, [Paper Link](https://ieeexplore.ieee.org/abstract/document/10478151). <br> 
[3] K. Kuwaranancharoen, L. Xin and S. Sundaram, *Byzantine-Resilient Distributed Optimization of Multi-Dimensional Functions*, in IEEE American Control Conference (ACC), 2020, pp. 4399-4404 [Paper Link](https://ieeexplore.ieee.org/abstract/document/9147396). <br>
#### Other References
[4] C. Fang, Z. Yang, and W. U. Bajwa, *BRIDGE: Byzantine-Resilient Decentralized Gradient Descent*, IEEE Transactions on Signal and Information Processing over Networks, 8 (2022), pp. 610–626. <br>
[5] W. Fu, Q. Ma, J. Qin, and Y. Kang, *Resilient Consensus-Based Distributed Optimization under Deception Attacks*, International Journal of Robust and Nonlinear Control, 31 (2021), pp. 1803–1816. <br>
[6] L. Su and N. Vaidya, *Byzantine Multi-Agent Optimization: Part I*, 2015, https://arxiv.org/abs/1506.04681. <br>
[7] L. Su and N. H. Vaidya, *Byzantine-Resilient Multiagent Optimization*, IEEE Transactions on Automatic Control, 66 (2020), pp. 2227–2233. <br>
[8] S. Sundaram and B. Gharesifard, *Distributed Optimization under Adversarial Nodes*, IEEE Transactions on Automatic Control, 64 (2018), pp. 1063–1076. <br>
[9] C. Zhao, J. He, and Q.-G. Wang, *Resilient Distributed Optimization Algorithm against Adversarial Attacks*, IEEE Transactions on Automatic Control, 65 (2019), pp. 4308–4315. <br>
[10] W. Abbas, M. Shabbir, J. Li, and X. Koutsoukos, *Resilient Distributed Vector Consensus using Centerpoint*, Automatica, 136 (2022), p. 110046. <br>
[11] H. Park and S. A. Hutchinson, *Fault-Tolerant Rendezvous of Multirobot Systems*, IEEE transactions on robotics, 33 (2017), pp. 565–582. <br>
