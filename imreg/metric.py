""" A collection of image similarity metrics. """

import collections

import numpy as np

# Container for registration methods:

Method = collections.namedtuple('method', 'jacobian error update')

# ==============================================================================
# Forwards additive:
# ==============================================================================


def forwardsAdditiveJacobian(image, template, model, p):
    """
    Computes the jacobian dP/dE.

    Parameters
    ----------
    model: deformation model
        A particular deformation model.
    warpedImage: nd-array
        Input image after warping.
    p : optional list
        Current warp parameters

    Returns
    -------
    jacobian: nd-array
        A jacobain matrix. (m x n)
            | where: m = number of image pixels,
            |        p = number of parameters.
    """

    grad = np.gradient(image)
    dIx = grad[1].flatten()
    dIy = grad[0].flatten()

    dPx, dPy = model.jacobian(template.coords, p)

    J = np.zeros_like(dPx)
    for index in range(0, dPx.shape[1]):
        J[:, index] = (dPx[:, index] * dIx) + (dPy[:, index] * dIy)
    return J


def forwardsAdditiveError(image, template):
    """ Compute the forwards additive error """
    return (template - image).flatten()


def forwardsAdditiveUpdate(p, deltaP, model=None):
    """ Compute the forwards additive error """
    return p + deltaP

# Define the forwards additive approach:

forwardsAdditive = Method(
    forwardsAdditiveJacobian,
    forwardsAdditiveError,
    forwardsAdditiveUpdate
    )

# ==============================================================================
# Forwards compositional.
# ==============================================================================


def forwardsCompositionalError(image, template):
    """ Compute the forwards additive error """
    return (template - image).flatten()


def forwardsCompositionalUpdate(p, deltaP, tform):
    """ Compute the forwards additive error """
    return tform.vectorForm(np.dot(tform.matrixForm(p), tform.matrixForm(deltaP)))

forwardsCompositional = Method(
    forwardsAdditiveJacobian,
    forwardsCompositionalError,
    forwardsCompositionalUpdate
    )


# ==============================================================================
# Inverse compositional.
# ==============================================================================


def inverseCompositionalJacobian(image, template, model, p):
    """
    Computes the jacobian dP/dE.

    Parameters
    ----------
    model: deformation model
        A particular deformation model.
    warpedImage: nd-array
        Input image after warping.
    p : optional list
        Current warp parameters

    Returns
    -------
    jacobian: nd-array
        A jacobain matrix. (m x n)
            | where: m = number of image pixels,
            |        p = number of parameters.
    """

    grad = np.gradient(template.data)
    dIx = grad[1].flatten()
    dIy = grad[0].flatten()

    dPx, dPy = model.jacobian(template.coords, p)

    J = np.zeros_like(dPx)
    for index in range(0, dPx.shape[1]):
        J[:, index] = (dPx[:, index] * dIx) + (dPy[:, index] * dIy)
    return J


def inverseCompositionalError(image, template):
    """ Compute the inverse additive error """
    return (image - template).flatten()


def inverseCompositionalUpdate(p, deltaP, tform):
    """ Compute the inverse compositional error """

    T = np.dot(tform.matrixForm(p), np.linalg.inv(tform.matrixForm(deltaP)))

    return tform.vectorForm(T)


inverseCompositional = Method(
    inverseCompositionalJacobian,
    inverseCompositionalError,
    inverseCompositionalUpdate
    )

