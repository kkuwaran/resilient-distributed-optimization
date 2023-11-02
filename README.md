# resilient-distributed-optimization
### Resilient Distributed Optimization Algorithms/Algorithmic Framework 
In this repository, we focus on addressing Byzantine-resilient distributed optimization problems. Our work revolves around the implementation of the **RE**silient **D**istributed **GR**adient-descent **A**lgorithmic **F**ramework **(REDGRAF)** [1], which encompasses various resilient algorithms. These algorithms are designed to overcome Byzantine failures, ensuring robust distributed optimization. 

Implemented Algorithms:
-  **Simultaneous Distance-MixMax Filtering Dynamics (SDMMFD)** [2, 3]
-  **Simultaneous Distance Filtering Dynamics (SDFD)** [2]
-  **Coordinate-wise Trimmed Mean (CWTM)** [4-9]
-  **Coordinate-wise Median (CWMed)** [4]
-  **Resilient Vector Optimization (RVO)** [10, 11]

### Experimentation and Analysis
We conduct comprehensive experiments to evaluate the performance of each algorithm. Our experiments include two key components:
- **Quadratic Functions Experiment:** We examine how these algorithms perform in optimizing distributed quadratic functions. This experiment helps us understand their convergence and efficiency in a controlled setting.
- **Banknote Authentication Experiment (Classification Task):** We delve into a real-world classification problem: authenticating banknotes. By applying our resilient algorithms to this task, we assess their effectiveness in solving practical optimization challenges.

### Dependency Plots
To gain insights into the behavior of these algorithms, we provide plots that showcase their dependency on the constant step-size. These plots specifically illustrate the following key quantities:
- **Convergence Rate:** How quickly do the algorithms converge to a region containing the optimal solution?
- **Convergence Radius:** What is the reach of the algorithms in terms of convergence?
- **Approximate Consensus Diameter:** How closely do the algorithms approach a consensus state?

### Folder Structure
- Folder **modules**
  - **main_script.py:** the main script used to configure and run the experiments shown in [1]
  - **experiments.py:** execute resilient algorithms, get results, and plot
  - **algorithmic_framework.py:** implement REDGRAF
  - **resilient_algorithms.py:** implement resilient algorithms
  - **adversaries.py:** initialize Byzantine agents, and determine their behavior
  - **objective_functions.py:** initialize (distributed) quadratic functions, and banknote authentication task
  - **topology_generation.py:** initialize network topology
  - **logistic_regression.py:** helping functions for Logistic Regression
  - **centerpoint.py:** helping functions for executing RVO (from [11])
  - **dependency_plots.py:** plot figures regarding the convergence rate, convergence radius, and consensus diameter shown in [1]
- Folder **figures**
  - **variables_dependency:** store figures generated from dependency_plots.py
  - **exp_quadratic:** store figures generated from running quadratic functions experiment
  - **exp_banknote:** store figures generated from running banknote authentication experiment
- Folder **datasets**
  - **BankNote_Authentication.csv:** store dataset used for banknote authentication task
- Folder **tests**
  - **algorithmic_framework_test.py:** script for testing algorithmic_framework.py
  - **resilient_algorithms_test.py:** script for testing resilient_algorithms.py
  - **adversaries_test.py:** script for testing adversaries.py
  - **dataset_functions_test.py:** script for testing DecentralizedDataset class in objective_functions.py
  - **quadratic_functions_test.py:** script for testing DecentralizedQuadratic class in objective_functions.py
  - **topology_generation_test.py:** script for testing topology_generation.py
  - **logistic_regression_test.py:** script for testing logistic_regression.py

### References
#### Our Relevant Works
[1] K. Kuwaranancharoen and S. Sundaram, *On the Geometric Convergence of Byzantine-Resilient Distributed Optimization Algorithms*, 2023, [Paper Link](https://arxiv.org/abs/2305.10810). <br>
[2] K. Kuwaranancharoen, L. Xin, and S. Sundaram, *Scalable Distributed Optimization of Multi-Dimensional Functions Despite Byzantine Adversaries*, 2022, [Paper Link](https://arxiv.org/abs/2003.09038). <br>
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
