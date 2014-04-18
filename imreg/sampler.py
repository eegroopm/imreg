""" A collection of samplers"""

import numpy as np
import scipy.ndimage as nd

try:
    from imreg import interpolation
except ImportError as error:
    # Attempt autocompilation.
    import pyximport
    pyximport.install()
    from imreg import _interpolation as interpolation


def nearest(image, warp):
    """
    Nearest-neighbour interpolation.

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

    result = np.zeros_like(warp[0], dtype=np.float64)

    interpolation.nearest(
        warp.astype(np.float64),
        image.astype(np.float64),
        result
        )

    return result


def bilinear(image, warp):
    """
    Bilinear interpolation.

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

    result = np.zeros_like(warp[0], dtype=np.float64)

    interpolation.bilinear(
        warp.astype(np.float64),
        image.astype(np.float64),
        result
        )

    return result


def spline(image, warp):
    """
    Spline interpolation.

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

    return nd.map_coordinates(
        image,
        warp,
        order=3,
        mode='nearest'
        )
