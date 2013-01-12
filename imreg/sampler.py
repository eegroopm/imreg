""" A collection of samplers"""

import numpy as np
import scipy.ndimage as nd

import interpolation

# Configuration for the extrapolation mode and fill value.
EXTRAPOLATION_MODE = 'c'
EXTRAPOLATION_CVALUE = 0.0


class Sampler(object):
    """
    Abstract sampler

    Attributes
    ----------
    METRIC : string
        The type of similarity sampler being used.
    DESCRIPTION : string
        A meaningful description of the sampler used, with references where
        appropriate.
    """

    METHOD = None
    DESCRIPTION = None

    def __init__(self, coordinates):

        self.coordinates = coordinates

    def f(self, array, warp):
        """
        A sampling function, responsible for returning a sampled set of values
        from the given array.

        Parameters
        ----------
        array: nd-array
            Input array for sampling.
        warp: nd-array
            Deformation coordinates.

        Returns
        -------
        sample: nd-array
           Sampled array data.
        """

        if self.coordinates is None:
            raise ValueError('Appropriately defined coordinates not provided.')

        i = self.coordinates.tensor[0] + warp[0]
        j = self.coordinates.tensor[1] + warp[1]

        packedCoords = i.reshape(1, i.size), j.reshape(1, j.size)

        return self.sample(array, np.vstack(packedCoords))

    def sample(self, array, coords):
        """
        The sampling function - provided by the specialized samplers.
        """
        return None

    def __str__(self):
        return 'Method: {0} \n {1}'.format(
            self.METHOD,
            self.DESCRIPTION
            )


class Nearest(Sampler):

    METHOD='Nearest Neighbour (NN)'

    DESCRIPTION="""
        Given coordinate in the array nearest neighbour sampling simply rounds
        coordinates points:
            f(I; i,j) = I( round(i), round(j))
        """

    def __init__(self, coordinates):
        Sampler.__init__(self, coordinates)


    def f(self, array, warp):
        """
        A sampling function, responsible for returning a sampled set of values
        from the given array.

        Parameters
        ----------
        array: nd-array
            Input array for sampling.
        warp: nd-array
            Deformation coordinates.

        Returns
        -------
        sample: nd-array
           Sampled array data.
        """

        if self.coordinates is None:
            raise ValueError('Appropriately defined coordinates not provided.')

        result = np.zeros_like(warp[0], dtype=np.float64)

        interpolation.nearest(warp.astype(np.float64), array.astype(np.float64), result)

        return result.flatten()


class Bilinear(Sampler):

    METHOD = 'Bilinear (BL)'

    DESCRIPTION = """
        Given a coordinate in the array a linear interpolation is performed
        between 4 (2x2) nearest values.
        """

    def __init__(self, coordinates):
        Sampler.__init__(self, coordinates)

    def f(self, array, warp):
        """
        A sampling function, responsible for returning a sampled set of values
        from the given array.

        Parameters
        ----------
        array: nd-array
            Input array for sampling.
        warp: nd-array
            Deformation coordinates.

        Returns
        -------
        sample: nd-array
           Sampled array data.
        """

        if self.coordinates is None:
            raise ValueError('Appropriately defined coordinates not provided.')

        result = np.zeros_like(warp[0], dtype=np.float64)

        interpolation.bilinear(warp.astype(np.float64), array.astype(np.float64), result)

        return result.flatten()


class Spline(Sampler):

    METHOD = 'nd-image spline sampler (SR)'

    DESCRIPTION = s"""
        Refer to the documentation for the ndimage map_coordinates function.

        http://docs.scipy.org/doc/scipy/reference/generated/
            scipy.ndimage.interpolation.map_coordinates.html
        """

    def __init__(self, coordinates):
        Sampler.__init__(self, coordinates)

    def f(self, array, warp):
        """
        A sampling function, responsible for returning a sampled set of values
        from the given array.

        Parameters
        ----------
        array: nd-array
            Input array for sampling.
        warp: nd-array
            Deformation coordinates.

        Returns
        -------
        sample: nd-array
           Sampled array data.
        """

        if self.coordinates is None:
            raise ValueError('Appropriately defined coordinates not provided.')

        return nd.map_coordinates(
            array,
            warp,
            order=2,
            mode='nearest'
            ).flatten()
