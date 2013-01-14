import numpy as np

from imreg import model, register, sampler

import scipy.misc as misc
import scipy.ndimage as nd


def test_shift():
    """
    Asserts that the feature point alignment error is sufficiently small.
    """

    # Form a dummy coordinate class.
    coords = model.Coordinates(
        [0, 10, 0, 10]
        )

    # Form corresponding feature sets.
    p0 = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4, 2)
    p1 = p0 + 2.0

    shift = model.Shift()

    _parameters, error = shift.fit(p0, p1)

    print _parameters

    # Assert that the alignment error is small.
    assert error <= 1.0, "Unexpected large alignment error : {} grid units".format(error)


def test_affine():
    """
    Asserts that the feature point alignment error is sufficiently small.
    """

    # Form a dummy coordinate class.
    coords = model.Coordinates(
        [0, 10, 0, 10]
        )

    # Form corresponding feature sets.
    p0 = np.array([0, 0, 0, 1, 1, 0, 1, 1]).reshape(4,2)
    p1 = p0 + 2.0

    affine = model.Affine()

    _parameters, error = affine.fit(p0, p1)

    # Assert that the alignment error is small.
    assert error <= 1.0, "Unexpected large alignment error : {} grid units".format(error)
