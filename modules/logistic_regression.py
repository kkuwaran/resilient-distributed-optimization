# Logistic Regression Helper Functions from 
# https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/linear_model/_logistic.py#L46

import numpy as np

from scipy import sparse, special
from sklearn.utils._logistic_sigmoid import _log_logistic_sigmoid
from sklearn.utils import check_array


class Logistic_Regression:
    def __init__(self, X, y, alpha):
        '''Functions for executing Logistic Regression 
        ========== Inputs ==========
        X - ndarray (n_samples, n_features): Features Matrix
        y - ndarray (n_samples, ): Labels Vector containing {-1, 1}
        alpha - positive scalar: Regularization Parameter
        '''

        self.X = X
        self.y = y
        self.alpha = alpha

    # ============================== Loss and Gradient Calculation ==============================

    ###@_deprecate_positional_args
    def safe_sparse_dot(self, a, b, *, dense_output=False):
        '''Dot product that handle the sparse matrix case correctly
        ========== Inputs ==========
        a - array or sparse matrix
        b - array or sparse matrix
        dense_output - boolean, (default=False)
        When False, "a" and "b" both being sparse will yield sparse output.
        When True, output will always be a dense array.
        ========== Output ==========
        dot_product - array or sparse matrix: sparse if "a" and "b" are sparse and "dense_output=False".
        '''

        if a.ndim > 2 or b.ndim > 2:
            if sparse.issparse(a):
                # sparse is always 2D. Implies b is 3D+
                # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
                b_ = np.rollaxis(b, -2)
                b_2d = b_.reshape((b.shape[-2], -1))
                ret = a @ b_2d
                ret = ret.reshape(a.shape[0], *b_.shape[1:])
            elif sparse.issparse(b):
                # sparse is always 2D. Implies a is 3D+
                # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
                a_2d = a.reshape(-1, a.shape[-1])
                ret = a_2d @ b
                ret = ret.reshape(*a.shape[:-1], b.shape[1])
            else:
                ret = np.dot(a, b)
        else:
            ret = a @ b
        
        if (sparse.issparse(a) and sparse.issparse(b) and dense_output and hasattr(ret, "toarray")):
            return ret.toarray()
        return ret


    def log_logistic(self, X, out=None):
        '''Compute the log of the logistic function, "log(1 / (1 + e ** -x))"
        This implementation is numerically stable because it splits positive and negative values:
          -log(1 + exp(-x_i)) if x_i > 0; x_i - log(1 + exp(x_i)) if x_i <= 0
        For the ordinary logistic function, use "scipy.special.expit".
        ========== Inputs ==========
        self.X - array-like, shape (M, N) or (M, ): Argument to the logistic function
        out - array-like, shape: (M, N) or (M, ), optional: Preallocated output array.
        ========== Output ==========
        out - array, shape (M, N) or (M, ): Log of the logistic function evaluated at every point in x
        ========== Notes ========== 
        See the blog post describing this implementation:
        http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
        '''
    
        is_1d = X.ndim == 1
        X = np.atleast_2d(X)
        X = check_array(X, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        if out is None:
            out = np.empty_like(X)
        
        _log_logistic_sigmoid(n_samples, n_features, X, out)
        
        if is_1d:
            return np.squeeze(out)
        return out
  

    def intercept_dot(self, w):
        '''Computes y * np.dot(X, w). It takes into consideration if the intercept should be fit or not.
        ========== Inputs ==========
        w - ndarray of shape (n_features,) or (n_features + 1,): Coefficient vector.
        self.X - {array-like, sparse matrix} of shape (n_samples, n_features): Training data.
        self.y - ndarray of shape (n_samples,): Array of labels.
        ========== Outputs ==========
        w - ndarray of shape (n_features,): Coefficient vector without the 
        intercept weight (w[-1]) if the intercept should be fit. Unchanged otherwise.
        c - float: The intercept.
        yz - float: y * np.dot(X, w).
        '''

        c = 0.
        if w.size == self.X.shape[1] + 1:
            c = w[-1]
            w = w[:-1]
        z = self.safe_sparse_dot(self.X, w) + c
        #z = np.dot(self.X, w) + c
        yz = self.y * z
        return w, c, yz


    def logistic_loss_and_grad(self, w):
        '''Computes the logistic loss and gradient
        ========== Inputs ==========
        w - ndarray of shape (n_features,) or (n_features + 1,): Coefficient vector.
        self.X - {array-like, sparse matrix} of shape (n_samples, n_features): Training data.
        self.y - ndarray of shape (n_samples,): Array of labels. *** Convert Labels to [-1,1]
        self.alpha - float: Regularization parameter. alpha is equal to 1 / C.
        ========== Outputs ==========
        out - float: Logistic loss.
        grad - ndarray of shape (n_features,) or (n_features + 1,): Logistic gradient.
        '''

        n_samples, n_features = self.X.shape
        grad = np.empty_like(w)
        
        w, c, yz = self.intercept_dot(w)
        
        # Logistic loss is the negative of the log of the logistic function.
        out = -np.sum(self.log_logistic(yz)) + .5 * self.alpha * np.dot(w, w)
        #out = -np.sum(special.expit(yz)) + .5 * self.alpha * np.dot(w, w)
        
        z = special.expit(yz)
        z0 = (z - 1) * self.y
        
        grad[:n_features] = self.safe_sparse_dot(self.X.T, z0) + self.alpha * w
        #grad[:n_features] = np.dot(self.X.T, z0) + self.alpha * w
        
        # Case where we fit the intercept.
        if grad.shape[0] > n_features:
            grad[-1] = z0.sum()
        return out, grad