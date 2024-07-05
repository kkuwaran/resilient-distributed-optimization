import sys
import numpy as np

sys.path.insert(0, "modules/utilities")
from logistic_regression import Logistic_Regression


### Test Function: logistic_loss_and_grad
print("===== Test Function: logistic_loss_and_grad =====")
np.random.seed(0)
np.set_printoptions(precision=4)

# Parameters Set-up
n_features = 3
n_samples = 5
alpha = 1.0

# Inputs Construction
w = np.random.rand(n_features+1)
X = np.random.rand(n_samples, n_features)
y = 2 * np.random.randint(2, size=n_samples) - 1  # Convert to Vector of {-1, 1}
print("Features Matrix: \n", X)
print("Labels Vector:", y)
print("Weight Vector:", w)
print("Regularization Parameter:", alpha, end="\n\n")

# Caculation
logistic = Logistic_Regression(X, y, alpha)
loss, grad = logistic.logistic_loss_and_grad(w)
print("Loss:", loss)
print("Gradient:", grad)