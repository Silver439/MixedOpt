import numpy as np

from .optimization_problem import OptimizationProblem


class Ackley15I(OptimizationProblem):
    """Ackley function

    .. math::
        f(x_1,\\ldots,x_n) = -20\\exp\\left( -0.2 \\sqrt{\\frac{1}{n} \
        \\sum_{j=1}^n x_j^2} \\right) -\\exp \\left( \\frac{1}{n} \
        \\sum{j=1}^n \\cos(2 \\pi x_j) \\right) + 20 - e

    subject to

    .. math::
        -15 \\leq x_i \\leq 20

    Global optimum: :math:`f(0,0,...,0)=0`

    :ivar dim: Number of dimensions
    :ivar lb: Lower variable bounds
    :ivar ub: Upper variable bounds
    :ivar int_var: Integer variables
    :ivar cont_var: Continuous variables
    :ivar min: Global minimum value
    :ivar minimum: Global minimizer
    :ivar info: String with problem info
    """

    def __init__(self, dim=20):
        self.dim = dim
        self.min = 0
        self.minimum = np.zeros(dim)
        self.lb = 0 * np.ones(dim)
        self.lb[-5:] = -1
        self.ub = 1 * np.ones(dim)
        self.int_var = np.arange(0, 15)
        self.cat_var = np.array([])
        self.cont_var = np.arange(15, 20)
        self.info = str(dim) + "-dimensional Ackley function \n" + "Global optimum: f(0,0,...,0) = 0"

    def eval(self, x):
        """Evaluate the Ackley function  at x

        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: float
        """
        self.__check_input__(x)
        d = float(self.dim)
        return (
            -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / d))
            - np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
            + 20
            + np.exp(1)
        )
    