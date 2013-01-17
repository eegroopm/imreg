""" A collection of deformation models. """

import numpy as np
import scipy.signal as signal


class Coordinates(object):
    """
    Container for grid coordinates.

    Attributes
    ----------
    domain : nd-array
        Domain of the coordinate system.
    tensor : nd-array
        Grid coordinates.
    homogenous : nd-array
        `Homogenous` coordinate system representation of grid coordinates.
    """

    def __init__(self, domain, tensor=None):

        self.domain = domain

        if tensor is None:
            self.tensor = np.mgrid[domain[0]:domain[1], domain[2]:domain[3]]
        else:
            self.tensor = tensor

        self.homogenous = np.zeros((3, self.tensor[0].size))
        self.homogenous[0] = self.tensor[1].flatten()
        self.homogenous[1] = self.tensor[0].flatten()
        self.homogenous[2] = 1.0

    @staticmethod
    def fromTensor(tensor):
        domain = [
            tensor[1].min(),
            tensor[1].max(),
            tensor[0].min(),
            tensor[0].max()
            ]
        return Coordinates(domain, tensor)

    @property
    def xy(self):
        return self.homogenous[0], self.homogenous[1]

    @property
    def bbox(self):
        tl = self.homogenous[0][0], self.homogenous[1][0]
        tr = self.homogenous[0][-1], self.homogenous[1][0]
        bl = self.homogenous[0][0], self.homogenous[1][-1]
        br = self.homogenous[0][-1], self.homogenous[1][-1]
        return (tl, tr, bl, br)


class Shift(object):
    """
    Applies the shift coordinate transformation. Follows the derivations
    shown in:

    S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
    Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
    """

    def __call__(self, coords, parameters):
        return Coordinates.fromTensor(self.transform(parameters, coords))

    @property
    def identity(self):
        return np.zeros(2)

    def transform(self, parameters, coords):
        """
        A "shift" transformation of coordinates.

        Parameters
        ----------
        parameters: nd-array
            Model parameters.

        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """

        T = np.eye(3, 3)
        T[0, 2] = parameters[0]
        T[1, 2] = parameters[1]

        displacement = np.dot(T, coords.homogenous)

        shape = coords.tensor[0].shape

        return np.array(
            [displacement[1].reshape(shape), displacement[0].reshape(shape)]
            )

    def jacobian(self, coords, p=None):
        """
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((coords.tensor[0].size, 2))
        dy = np.zeros((coords.tensor[0].size, 2))

        dx[:, 0] = 1
        dy[:, 1] = 1

        return (dx, dy)


class Affine(object):
    """
    Applies the affine coordinate transformation. Follows the derivations
    shown in:

    S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
    Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
    """

    def __call__(self, coords, parameters):
        return Coordinates.fromTensor(self.transform(parameters, coords))

    @property
    def identity(self):
        return np.zeros(6)

    def transform(self, p, coords):
        """
        An "affine" transformation of coordinates.

        Parameters
        ----------
        parameters: nd-array
            Model parameters.

        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """

        T = np.array([
                      [p[0] + 1.0, p[2],       p[4]],
                      [p[1],       p[3] + 1.0, p[5]],
                      [0,          0,          1]
                      ])

        displacement = np.dot(T, coords.homogenous)

        shape = coords.tensor[0].shape

        return np.array(
            [displacement[1].reshape(shape), displacement[0].reshape(shape)]
            )

    def jacobian(self, coords, p=None):
        """"
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((coords.tensor[0].size, 6))
        dy = np.zeros((coords.tensor[0].size, 6))

        dx[:, 0] = coords.tensor[1].flatten()
        dx[:, 2] = coords.tensor[0].flatten()
        dx[:, 4] = 1.0

        dy[:, 1] = coords.tensor[1].flatten()
        dy[:, 3] = coords.tensor[0].flatten()
        dy[:, 5] = 1.0

        return (dx, dy)


class Projective(object):
    """
    Applies the projective coordinate transformation. Follows the derivations
    shown in:

    S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
    Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
    """

    def __call__(self, coords, parameters):
        return Coordinates.fromTensor(self.transform(parameters, coords))

    @property
    def identity(self):
        return np.zeros(9)

    def transform(self, p, coords):
        """
        An "projective" transformation of coordinates.

        Parameters
        ----------
        parameters: nd-array
            Model parameters.

        Returns
        -------
        coords: nd-array
           Deformation coordinates.
        """

        T = np.array([
                [p[0] + 1.0, p[2],       p[4]],
                [p[1],       p[3] + 1.0, p[5]],
                [p[6],       p[7],       p[8] + 1.0]
                ])

        displacement = np.dot(np.linalg.inv(T), coords.homogenous)

        shape = coords.tensor[0].shape

        return np.array(
            [displacement[1].reshape(shape), displacement[0].reshape(shape)]
            )

    def jacobian(self, coords, p):
        """"
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((coords.tensor[0].size, 9))
        dy = np.zeros((coords.tensor[0].size, 9))

        x = coords.tensor[1].flatten()
        y = coords.tensor[0].flatten()

        dx[:, 0] = x / (p[6] * x + p[7] * y + p[8] + 1)
        dx[:, 2] = y / (p[6] * x + p[7] * y + p[8] + 1)
        dx[:, 4] = 1.0 / (p[6] * x + p[7] * y + p[8] + 1)
        dx[:, 6] = x * (p[0] * x + p[2] * y + p[4] + x) / (p[6] * x + p[7] * y + p[8] + 1)**2
        dx[:, 7] = y * (p[0] * x + p[2] * y + p[4] + x) / (p[6] * x + p[7] * y + p[8] + 1)**2
        dx[:, 8] = 1.0 * (p[0] * x + p[2] * y + p[4] + x) / (p[6] * x + p[7] * y + p[8] + 1)**2

        dy[:, 1] = x / (p[6] * x + p[7] * y + p[8] + 1)
        dy[:, 3] = y / (p[6] * x + p[7] * y + p[8] + 1)
        dy[:, 5] = 1.0 / (p[6] * x + p[7] * y + p[8] + 1)
        dy[:, 6] = x * (p[1] * x + p[3] * y + p[5] + y) / (p[6] * x + p[7] * y + p[8] + 1)**2
        dy[:, 7] = y * (p[1] * x + p[3] * y + p[5] + y) / (p[6] * x + p[7] * y + p[8] + 1)**2
        dy[:, 8] = 1.0 * (p[1] * x + p[3] * y + p[5] + y) / (p[6] * x + p[7] * y + p[8] + 1)**2

        return (dx, dy)
