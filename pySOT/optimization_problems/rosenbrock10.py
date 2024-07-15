import numpy as np

from .optimization_problem import OptimizationProblem


class Rosenbrock10(OptimizationProblem):
    """Rosenbrock function

    .. math::
        f(x_1,\\ldots,x_n) = \\sum_{j=1}^{n-1} \
        \\left( 100(x_j^2-x_{j+1})^2 + (1-x_j)^2 \\right)

    subject to

    .. math::
        -2.048 \\leq x_i \\leq 2.048

    Global optimum: :math:`f(1,1,...,1)=0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=10):
        self.dim = dim
        self.min = 0
        self.minimum = np.ones(dim)
        self.lb = np.array([-0.5,-0.5,-0.5,-0.5,-0.5,-1,-1,-1,-1,-1])
        self.ub = np.array([4.5,4.5,4.5,4.5,4.5,1,1,1,1,1])
        self.int_var = np.array([])
        self.cont_var = np.arange(0, dim)
        self.info = str(dim) + "-dimensional Rosenbrock function \n" + "Global optimum: f(3,3,..., 0.5,0.5) = 0"

    def eval(self, X):
        """Evaluate the Rosenbrock function at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(X)
        x = X.copy()
        x[:5] = np.round(x[:5]-2)
        x[5:10] = x[5:10]*2
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i] ** 2 - x[i + 1]) ** 2 + (x[i] - 1) ** 2
        return total
