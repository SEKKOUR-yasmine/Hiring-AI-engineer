import numpy as np


class Kernel:
    def __add__(self, other):
        """
        Add two kernel functions.

        Parameters:
        - other: Another kernel function.

        Returns:
        The sum of the two kernel functions.
        """
        return SumKernel(self, other)

    def __mul__(self, other):
        """
        Multiply two kernel functions.

        Parameters:
        - other: Another kernel function.

        Returns:
        The product of the two kernel functions.
        """
        return ProductKernel(self, other)


class SumKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        """
        Initialize a sum kernel with two kernels.

        Parameters:
        - kernel1: The first kernel.
        - kernel2: The second kernel.
        """
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def compute(self, x1, x2):
        """
        Compute the sum of two kernel functions.

        Parameters:
        - x1: The first point.
        - x2: The second point.

        Returns:
        The computed sum of the two kernel functions.
        """
        return self.kernel1.compute(x1, x2) + self.kernel2.compute(x1, x2)


class ProductKernel(Kernel):
    def __init__(self, kernel1, kernel2):
        """
        Initialize a product kernel with two kernels.

        Parameters:
        - kernel1: The first kernel.
        - kernel2: The second kernel.
        """
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def compute(self, x1, x2):
        """
        Compute the product of two kernel functions.

        Parameters:
        - x1: The first point.
        - x2: The second point.

        Returns:
        The computed product of the two kernel functions.
        """
        return self.kernel1.compute(x1, x2) * self.kernel2.compute(x1, x2)


class GaussianKernel(Kernel):
    def __init__(self, length_scale=1.0):
        """
        Initialize the Gaussian kernel with a specified length scale.

        Parameters:
        - length_scale (float): The length scale of the Gaussian kernel.
        """
        self.length_scale = length_scale

    def compute(self, x1, x2):
        """
        Compute the Gaussian kernel between two points.

        Parameters:
        - x1: The first point.
        - x2: The second point.

        Returns:
        The computed Gaussian kernel value.
        """
        return np.exp(-0.5 * np.sum((x1 - x2) ** 2, axis=-1) / self.length_scale**2)


class RBFKernel(Kernel):
    def __init__(self, length_scale=1.0):
        """
        Initialize the Radial Basis Function (RBF) kernel with a specified length scale.

        Parameters:
        - length_scale (float): The length scale of the RBF kernel.
        """
        self.length_scale = length_scale

    def compute(self, x1, x2):
        """
        Compute the RBF kernel between two points.

        Parameters:
        - x1: The first point.
        - x2: The second point.

        Returns:
        The computed RBF kernel value.
        """
        return np.exp(
            -0.5 * np.sum((x1 - x2) ** 2, axis=-1) / (2 * self.length_scale**2)
        )


class RationalQuadraticKernel(Kernel):
    def __init__(self, alpha=1.0, length_scale=1.0):
        """
        Initialize the Rational Quadratic kernel with a specified alpha and length scale.

        Parameters:
        - alpha (float): The alpha parameter.
        - length_scale (float): The length scale of the kernel.
        """
        self.alpha = alpha
        self.length_scale = length_scale

    def compute(self, x1, x2):
        """
        Compute the Rational Quadratic kernel between two points.

        Parameters:
        - x1: The first point.
        - x2: The second point.

        Returns:
        The computed Rational Quadratic kernel value.
        """
        dist = np.sum((x1 - x2) ** 2, axis=-1)
        return (1 + dist / (2 * self.alpha * self.length_scale**2)) ** (-self.alpha)


class ExponentiatedKernelSineKernel(Kernel):
    def __init__(self, periodicity=1.0, length_scale=1.0):
        """
        Initialize the Exponentiated Kernel Sine kernel with a specified periodicity and length scale.

        Parameters:
        - periodicity (float): The periodicity of the kernel.
        - length_scale (float): The length scale of the kernel.
        """
        self.length_scale = length_scale
        self.periodicity = periodicity

    def compute(self, x1, x2):
        """
        Compute the Exponentiated Kernel Sine kernel between two points.

        Parameters:
        - x1: The first point.
        - x2: The second point.

        Returns:
        The computed Exponentiated Kernel Sine kernel value.
        """
        dist1 = np.sin(np.pi / self.periodicity * np.linalg.norm(x1 - x2, axis=-1)) ** 2
        dist2 = np.linalg.norm(x1 - x2, axis=-1) ** 2 / (2 * self.length_scale**2)
        return np.exp(-2 * dist1 / (self.periodicity**2)) * np.exp(-dist2)


class LocallyPeriodicKernel:
    def __init__(self, length_scale=1.0, periodicity=1.0, variance=1.0):
        """
        Initialize the Locally Periodic kernel with a specified length scale, periodicity, and variance.

        Parameters:
        - length_scale (float): The length scale of the kernel.
        - periodicity (float): The periodicity of the kernel.
        - variance (float): The variance of the kernel.
        """
        self.length_scale = length_scale
        self.periodicity = periodicity
        self.variance = variance

    def compute(self, x1, x2):
        """
        Compute the Locally Periodic kernel between two points.

        Parameters:
        - x1: The first point.
        - x2: The second point.

        Returns:
        The computed Locally Periodic kernel value.
        """
        per = ExponentiatedKernelSineKernel()
        K_per = per.compute(x1, x2)
        dist = np.sum((x1 - x2) ** 2, axis=-1)
        K_se = np.exp(-dist / self.length_scale**2)
        return K_per * K_se
