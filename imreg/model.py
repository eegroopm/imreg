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
            tensor[0].min(),
            tensor[0].max(),
            tensor[1].min(),
            tensor[1].max()
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


def warp(model, parameters, coords):
    """
    Computes the warp field given model parameters.

    Parameters
    ----------
    parameters: nd-array
        Model parameters.

    Returns
    -------
    warp: nd-array
       Deformation field.
    """

    displacement = model.transform(parameters, coords)

    # Approximation of the inverse (samplers work on inverse warps).
    return Coordinates.fromTensor(coords.tensor + displacement)


class Shift(object):

    MODEL='Shift (S)'

    DESCRIPTION="""
        Applies the shift coordinate transformation. Follows the derivations
        shown in:

        S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
        """

    def __init__(self):
        pass

    @property
    def identity(self):
        return np.zeros(2)

    @staticmethod
    def scale(p, factor):
        """
        Scales an shift transformation by a factor.

        Parameters
        ----------
        p: nd-array
            Model parameters.
        factor: float
            A scaling factor.

        Returns
        -------
        parameters: nd-array
            Model parameters.
        """

        pHat = p.copy()
        pHat *= factor
        return pHat

    def fit(self, p0, p1, lmatrix=False):
        """
        Estimates the best fit parameters that define a warp field, which
        deforms feature points p0 to p1.

        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).

        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        """

        parameters = p1.mean(axis=0) - p0.mean(axis=0)

        projP0 = p0 + parameters

        error = np.sqrt(
           (projP0[:,0] - p1[:,0])**2 + (projP0[:,1] - p1[:,1])**2
           ).sum()

        return -parameters, error

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
        T[0, 2] = -parameters[0]
        T[1, 2] = -parameters[1]

        displacement = np.dot(T, coords.homogenous) - \
            coords.homogenous

        shape = coords.tensor[0].shape

        return np.array( [ displacement[1].reshape(shape),
                           displacement[0].reshape(shape)
                         ]
                       )

    def jacobian(self, coords, p=None):
        """
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((coords.tensor[0].size, 2))
        dy = np.zeros((coords.tensor[0].size, 2))

        dx[:,0] = 1
        dy[:,1] = 1

        return (dx, dy)


class Affine(object):

    MODEL='Affine (A)'

    DESCRIPTION="""
        Applies the affine coordinate transformation. Follows the derivations
        shown in:

        S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
        """

    @property
    def identity(self):
        return np.zeros(6)


    @staticmethod
    def scale(p, factor):
        """
        Scales an affine transformation by a factor.

        Parameters
        ----------
        p: nd-array
            Model parameters.
        factor: float
            A scaling factor.

        Returns
        -------
        parameters: nd-array
            Model parameters.
        """

        pHat = p.copy()
        pHat[4:] *= factor
        return pHat


    def fit(self, p0, p1, lmatrix=False):
        """
        Estimates the best fit parameters that define a warp field, which
        deforms feature points p0 to p1.

        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).

        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        """

        # Solve: H*X = Y
        # ---------------------
        #          H = Y*inv(X)

        X = np.ones((3, len(p0)))
        X[0:2,:] = p0.T

        Y = np.ones((3, len(p0)))
        Y[0:2,:] = p1.T

        H = np.dot(Y, np.linalg.pinv(X))

        parameters = [
            H[0,0] - 1.0,
            H[1,0],
            H[0,1],
            H[1,1] - 1.0,
            H[0,2],
            H[1,2]
            ]

        projP0 = np.dot(H, X)[0:2,:].T

        error = np.sqrt(
           (projP0[:,0] - p1[:,0])**2 + (projP0[:,1] - p1[:,1])**2
           ).sum()

        return parameters, error


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
                      [p[0]+1.0, p[2],     p[4]],
                      [p[1],     p[3]+1.0, p[5]],
                      [0,         0,         1]
                      ])

        displacement = np.dot(np.linalg.inv(T), coords.homogenous) - \
            coords.homogenous

        shape = coords.tensor[0].shape

        return np.array( [ displacement[1].reshape(shape),
                           displacement[0].reshape(shape)
                         ]
                       )


    def jacobian(self, coords, p=None):
        """"
        Evaluates the derivative of deformation model with respect to the
        coordinates.
        """

        dx = np.zeros((coords.tensor[0].size, 6))
        dy = np.zeros((coords.tensor[0].size, 6))

        dx[:,0] = coords.tensor[1].flatten()
        dx[:,2] = coords.tensor[0].flatten()
        dx[:,4] = 1.0

        dy[:,1] = coords.tensor[1].flatten()
        dy[:,3] = coords.tensor[0].flatten()
        dy[:,5] = 1.0

        return (dx, dy)


class Projective(object):

    MODEL='Projective (P)'

    DESCRIPTION="""
        Applies the projective coordinate transformation. Follows the derivations
        shown in:

        S. Baker and I. Matthews. 2004. Lucas-Kanade 20 Years On: A
        Unifying Framework. Int. J. Comput. Vision 56, 3 (February 2004).
        """


    @property
    def identity(self):
        return np.zeros(9)


    def fit(self, p0, p1, lmatrix=False):
        """
        Estimates the best fit parameters that define a warp field, which
        deforms feature points p0 to p1.

        Parameters
        ----------
        p0: nd-array
            Image features (points).
        p1: nd-array
            Template features (points).

        Returns
        -------
        parameters: nd-array
            Model parameters.
        error: float
            Sum of RMS error between p1 and alinged p0.
        """

        # Solve: H*X = Y
        # ---------------------
        #          H = Y*inv(X)

        X = np.ones((3, len(p0)))
        X[0:2,:] = p0.T

        Y = np.ones((3, len(p0)))
        Y[0:2,:] = p1.T

        H = np.dot(Y, np.linalg.pinv(X))

        parameters = [
            H[0,0] - 1.0,
            H[1,0],
            H[0,1],
            H[1,1] - 1.0,
            H[0,2],
            H[1,2],
            H[2,0],
            H[2,1],
            H[2,2] - 1.0
            ]

        projP0 = np.dot(H, X)[0:2,:].T

        error = np.sqrt(
           (projP0[:,0] - p1[:,0])**2 + (projP0[:,1] - p1[:,1])**2
           ).sum()

        return parameters, error


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
                      [p[0]+1.0, p[2],     p[4]],
                      [p[1],     p[3]+1.0, p[5]],
                      [p[6],     p[7],     p[8]+1.0]
                      ])

        displacement = np.dot(np.linalg.inv(T), coords.homogenous) - \
            coords.homogenous

        shape = coords.tensor[0].shape

        return np.array( [ displacement[1].reshape(shape),
                           displacement[0].reshape(shape)
                         ]
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

        dx[:,0] = x / (p[6]*x + p[7]*y + p[8] + 1)
        dx[:,2] = y / (p[6]*x + p[7]*y + p[8] + 1)
        dx[:,4] = 1.0 / (p[6]*x + p[7]*y + p[8] + 1)
        dx[:,6] = x * (p[0]*x + p[2]*y + p[4] + x) / (p[6]*x + p[7]*y + p[8] + 1)**2
        dx[:,7] = y * (p[0]*x + p[2]*y + p[4] + x) / (p[6]*x + p[7]*y + p[8] + 1)**2
        dx[:,8] = 1.0 * (p[0]*x + p[2]*y + p[4] + x) / (p[6]*x + p[7]*y + p[8] + 1)**2

        dy[:,1] = x / (p[6]*x + p[7]*y + p[8] + 1)
        dy[:,3] = y / (p[6]*x + p[7]*y + p[8] + 1)
        dy[:,5] = 1.0 / (p[6]*x + p[7]*y + p[8] + 1)
        dy[:,6] = x * (p[1]*x + p[3]*y + p[5] + y) / (p[6]*x + p[7]*y + p[8] + 1)**2
        dy[:,7] = y * (p[1]*x + p[3]*y + p[5] + y) / (p[6]*x + p[7]*y + p[8] + 1)**2
        dy[:,8] = 1.0 * (p[1]*x + p[3]*y + p[5] + y) / (p[6]*x + p[7]*y + p[8] + 1)**2

        return (dx, dy)


    @staticmethod
    def scale(p, factor):
        """
        Scales an projective transformation by a factor.

        Derivation: If    Hx =  x^ ,
                    then SHx = Sx^ ,
                    where  S = [[s, 0, 0], [0, s, 0], [0, 0, 1]] .
                    Now   SH = S[[h00, h01, h02], [h10, h11, h12], [h20, h21, h22]]
                             =  [[s.h00, s.h01, s.h02], [s.h10, s.h11, s.h12], [h20, h21, h22]] .


        Parameters
        ----------
        p: nd-array
            Model parameters.
        factor: float
            A scaling factor.

        Returns
        -------
        parameters: nd-array
            Model parameters.
        """

        pHat = p.copy()
        pHat[0:6] *= factor
        return pHat
