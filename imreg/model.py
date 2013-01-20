""" A collection of deformation models. """

import numpy as np


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
        domain = [tensor[1].min(), tensor[1].max(), tensor[0].min(), tensor[0].max()]
        return Coordinates(domain, tensor)

    @property
    def xy(self):
        return self.homogenous[0], self.homogenous[1]


# ==============================================================================
# 2 - DOF
# ==============================================================================


class Shift(object):
    """
    Applies the shift coordinate transformation. Follows the derivations
    shown in:

    S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
    Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
    """

    def __call__(self, p, coords):
        return Coordinates.fromTensor(self.transform(p, coords))

    @property
    def identity(self):
        return np.zeros(2)

    def matrix(self, p):
        return np.array([
            (1.0, 0.0, p[0]),
            (0.0, 1.0, p[1]),
            (0.0, 0.0, 1.0)
            ])

    def vector(self, H):
        return H[0:2, 2]

    def transform(self, p, coords):
        """
        A "shift" transformation of coordinates.
        """

        displacement = np.dot(self.matrix(p), coords.homogenous)

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

        dx[:, 0] = 1.0
        dy[:, 1] = 1.0

        return (dx, dy)

# ==============================================================================
# 6 - DOF
# ==============================================================================


class Affine(object):
    """
    Applies the affine coordinate transformation. Follows the derivations
    shown in:

    S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
    Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
    """

    def __call__(self, p, coords):
        return Coordinates.fromTensor(self.transform(p, coords))

    @property
    def identity(self):
        return np.zeros(6)

    def matrix(self, p):
        return np.array([
            [p[0] + 1.0, p[2],       p[4]],
            [p[1],       p[3] + 1.0, p[5]],
            [0.0,        0.0,         1.0]
            ])

    def vector(self, H):
        return np.array(
            [H[0, 0] - 1.0,
             H[1, 0],
             H[0, 1],
             H[1, 1] - 1.0,
             H[0, 2],
             H[1, 2],
             ])

    def transform(self, p, coords):
        """
        An "affine" transformation of coordinates.
        """

        displacement = np.dot(self.matrix(p), coords.homogenous)
        shape = coords.tensor[0].shape

        return np.array(
            [displacement[1].reshape(shape), displacement[0].reshape(shape)]
            )

    def jacobian(self, coords, p=None):
        """"
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        x, y = coords.xy

        dx = np.zeros((x.size, 6))
        dy = np.zeros((x.size, 6))

        dx[:, 0] = x
        dx[:, 2] = y
        dx[:, 4] = 1.0

        dy[:, 1] = x
        dy[:, 3] = y
        dy[:, 5] = 1.0

        return (dx, dy)

# ==============================================================================
# 8 - DOF
# ==============================================================================


class Homography(object):
    """
    Applies the projective coordinate transformation. Follows the derivations
    shown in:

    S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
    Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
    """

    def __call__(self, p, coords):
        return Coordinates.fromTensor(self.transform(p, coords))

    @property
    def identity(self):
        return np.zeros(8)

    def matrix(self, p):
        return np.array([
            [p[0] + 1.0, p[3],       p[6]],
            [p[1],       p[4] + 1.0, p[7]],
            [p[2],       p[5],        1.0]
            ])

    def vector(self, H):
        return np.array(
            [H[0, 0] - 1.0,
             H[1, 0],
             H[2, 0],
             H[0, 1],
             H[1, 1] - 1.0,
             H[2, 1],
             H[0, 2],
             H[1, 2],
             ])

    def transform(self, p, coords):
        """
        An "projective" transformation of coordinates.
        """

        displacement = np.dot(self.matrix(p), coords.homogenous)
        shape = coords.tensor[0].shape
        return np.array(
            [displacement[1].reshape(shape), displacement[0].reshape(shape)]
            )

    def jacobian(self, coords, p):
        """"
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """
        x, y = coords.xy

        p0, p1, p2, p3, p4, p5, p6, p7 = p

        dx = np.zeros((x.size, 8))
        dy = np.zeros((x.size, 8))

        dx[:, 0] = x/(p2*x + p5*y + 1.0)
        dx[:, 1] = 0.0
        dx[:, 2] = -x*(p3*y + 1.0*p6 + x*(p0 + 1))/(p2*x + p5*y + 1.0)**2
        dx[:, 3] = y/(p2*x + p5*y + 1.0)
        dx[:, 4] = 0.0
        dx[:, 5] = -y*(p3*y + 1.0*p6 + x*(p0 + 1))/(p2*x + p5*y + 1.0)**2
        dx[:, 6] = 1.0/(p2*x + p5*y + 1.0)
        dx[:, 7] = 0.0

        dy[:, 0] = 0.0
        dy[:, 1] = x/(p2*x + p5*y + 1.0)
        dy[:, 2] = -x*(p1*x + 1.0*p7 + y*(p4 + 1))/(p2*x + p5*y + 1.0)**2
        dy[:, 3] = 0.0
        dy[:, 4] = y/(p2*x + p5*y + 1.0)
        dy[:, 5] = -y*(p1*x + 1.0*p7 + y*(p4 + 1))/(p2*x + p5*y + 1.0)**2
        dy[:, 6] = 0.0
        dy[:, 7] = 1.0/(p2*x + p5*y + 1.0)

        return (dx, dy)
