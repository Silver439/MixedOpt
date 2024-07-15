import numpy as np

from .optimization_problem import OptimizationProblem

class Func2I(OptimizationProblem):
    """Func2C is a mixed categorical and continuous function. The first 2 dimensions are categorical(but treat as integer),
    with possible 3 and 5 possible values respectively. The last 2 dimensions are continuous"""

    """
    Global minimum of this function is at
    x* = [1, 1, -0.0898/2, 0.7126/2]
    with f(x*) = -0.2063
    """
    problem_type = 'mixed'

    def __init__(self, dim=4):
        # Specifies the indices of the dimensions that are categorical and continuous, respectively
        self.dim = 4
        self.min = -0.2063
        self.minimum = np.array([1, 1, -0.0898/2, 0.7126/2])
        # Specfies the range for the continuous variables
        self.lb = np.array([-0.5,-0.5,-1, -1])
        self.ub = np.array([2.5,4.5,1, 1])
        self.lamda = 1e-6
        self.mean, self.std = None, None
        self.int_var = np.array([0, 1])
        self.cat_var = np.array([])
        self.cont_var = np.array([2, 3])
        self.cat_value = [(0,1,2),(0,1,2,3,4)]
        self.info = str(dim) + "-dimensional Func2C function \n" + "Global optimum: f(1, 1, -0.0898/2, 0.7126/2) = -0.2063"

    def eval(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        res = np.zeros((N, ))
        X_cat = X[:, self.int_var]
        X_cont = X[:, self.cont_var]
        X_cont = X_cont * 2

        for i, X in enumerate(X):
            if X_cat[i, 0] == 0:
                res[i] = myrosenbrock(X_cont[i, :])
            elif X_cat[i, 0] == 1:
                res[i] = mysixhumpcamp(X_cont[i, :])
            else:
                res[i] = mybeale(X_cont[i, :])

            if X_cat[i, 1] == 0:
                res[i] += myrosenbrock(X_cont[i, :])
            elif X_cat[i, 1] == 1:
                res[i] += mysixhumpcamp(X_cont[i, :])
            else:
                res[i] += mybeale(X_cont[i, :])
        res += self.lamda * np.random.rand(*res.shape)
        return res[0]

# =============================================================================
# Rosenbrock Function (f_min = 0)
# https://www.sfu.ca/~ssurjano/rosen.html
# =============================================================================
def myrosenbrock(X):
    X = np.asarray(X)
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:  # one observation
        x1 = X[0]
        x2 = X[1]
    else:  # multiple observations
        x1 = X[:, 0]
        x2 = X[:, 1]
    fx = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    return fx.reshape(-1, 1) / 300


# =============================================================================
#  Six-hump Camel Function (f_min = - 1.0316 )
#  https://www.sfu.ca/~ssurjano/camel6.html
# =============================================================================
def mysixhumpcamp(X):
    X = np.asarray(X)
    X = np.reshape(X, (-1, 2))
    if len(X.shape) == 1:
        x1 = X[0]
        x2 = X[1]
    else:
        x1 = X[:, 0]
        x2 = X[:, 1]
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2 ** 2) * x2 ** 2
    fval = term1 + term2 + term3
    return fval.reshape(-1, 1) / 10


# =============================================================================
# Beale function (f_min = 0)
# https://www.sfu.ca/~ssurjano/beale.html
# =============================================================================
def mybeale(X):
    X = np.asarray(X) / 2
    X = X.reshape((-1, 2))
    if len(X.shape) == 1:
        x1 = X[0] * 2
        x2 = X[1] * 2
    else:
        x1 = X[:, 0] * 2
        x2 = X[:, 1] * 2
    fval = (1.5 - x1 + x1 * x2) ** 2 + (2.25 - x1 + x1 * x2 ** 2) ** 2 + (
            2.625 - x1 + x1 * x2 ** 3) ** 2
    return fval.reshape(-1, 1) / 50