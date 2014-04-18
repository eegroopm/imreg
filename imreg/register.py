""" A top level registration module """

import numpy as np

import logging

from imreg import metric
from imreg import model
from imreg import sampler

# Setup a loger.
log = logging.getLogger('imreg.register')

REGISTRATION_STEP = """
================================================================================
iteration  : {0}
parameters : {1}
error      : {2}
================================================================================
"""

REGISTRATION_STOP = """
================================================================================
Optimization break, maximum number of bad iterations exceeded.
================================================================================
"""


class RegisterData(object):
    """
    Container for registration data.

    Attributes
    ----------
    data : nd-array
        The image registration image values.
    coords : nd-array, optional
        The grid coordinates.
    """

    def __init__(self, data, coords=None, features=None, spacing=1.0):

        self.data = data.astype(np.double)

        if not coords:
            self.coords = model.Coordinates(
                [0, data.shape[0], 0, data.shape[1]],
                )
        else:
            self.coords = coords


class optStep():
    """
    A container class for optimization steps.

    Attributes
    ----------
    error: float
        Normalised fitting error.
    p: nd-array
        Model parameters.
    deltaP: nd-array
        Model parameter update vector.
    decreasing: boolean.
        State of the error function at this point.
    """

    def __init__(self, error=None, p=None, deltaP=None, decreasing=None):
        self.error = error
        self.p = p
        self.deltaP = deltaP
        self.decreasing = decreasing


class Register(object):
    """
    A registration class for estimating the deformation model parameters that
    best solve:

    | :math:`f( W(I;p), T )`
    |
    | where:
    |    :math:`f`     : is a similarity metric.
    |    :math:`W(x;p)`: is a deformation model (defined by the parameter set p).
    |    :math:`I`     : is an input image (to be deformed).
    |    :math:`T`     : is a template (which is a deformed version of the input).

    Notes:
    ------

    Solved using a modified gradient descent algorithm.

    .. [0] Levernberg-Marquardt algorithm,
           http://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm

    Attributes
    ----------
    model: class
        A `deformation` model class definition.
    """

    MAX_ITER = 200
    MAX_BAD = 5

    def __deltaP(self, J, e, alpha, p=None):
        """
        Computes the parameter update.

        Parameters
        ----------
        J: nd-array
            The (dE/dP) the relationship between image differences and model
            parameters.
        e: float
            The evaluated similarity metric.
        alpha: float
            A dampening factor.
        p: nd-array or list of floats, optional

        Returns
        -------
        deltaP: nd-array
           The parameter update vector.
        """

        H = np.dot(J.T, J)

        H += np.diag(alpha * np.diagonal(H))

        return np.dot(np.linalg.inv(H), np.dot(J.T, e))

    def __dampening(self, alpha, decreasing):
        """
        Computes the adjusted dampening factor.

        Parameters
        ----------
        alpha: float
            The current dampening factor.
        decreasing: boolean
            Conditional on the decreasing error function.

        Returns
        -------
        alpha: float
           The adjusted dampening factor.
        """
        return alpha / 10. if decreasing else alpha * 10.

    def register(self,
            image,
            template,
            tform,
            sampler=sampler.bilinear,
            method=metric.forwardsAdditive,
            p=None,
            alpha=None,
            verbose=False
            ):
        """
        Computes the registration between the image and template.

        Parameters
        ----------
        image: nd-array
            The floating image.
        template: nd-array
            The target image.
        tform: deformation (class)
            The deformation model (shift, affine, projective)
        method: collection, optional.
            The registration method (defaults to FrowardsAdditive)
        p: list (or nd-array), optional.
            First guess at fitting parameters.
        alpha: float
            The dampening factor.
        verbose: boolean
            A debug flag for text status updates.

        Returns
        -------
        step: optimization step.
            The best optimization step (after convergence).
        search: list (of optimization steps)
            The set of optimization steps (good and bad)
        """

        p = tform.identity if p is None else p
        deltaP = np.zeros_like(p)

        # Dampening factor.
        alpha = alpha if alpha is not None else 1e-4

        # Variables used to implement a back-tracking algorithm.
        search = []
        badSteps = 0
        bestStep = None

        for itteration in range(0, self.MAX_ITER):

            # Compute the transformed coordinates.
            coords = tform(p, template.coords)

            # Sample to the template frame using the transformed coordinates.
            warpedImage = sampler(image.data, coords.tensor)

            # Evaluate the error metric.
            e = method.error(warpedImage, template.data)

            # Cache the optimization step.
            searchStep = optStep(
               error=np.abs(e).sum() / np.prod(image.data.shape),
               p=p.copy(),
               deltaP=deltaP.copy(),
               decreasing=True
               )

            # Update the current best step.
            bestStep = searchStep if bestStep is None else bestStep

            if verbose:
                log.warn(
                    REGISTRATION_STEP.format(
                        itteration,
                        ' '.join('{0:3.2f}'.format(param) for param in searchStep.p),
                        searchStep.error
                        )
                    )

            # Append the search step to the search.
            search.append(searchStep)

            if len(search) > 1:

                searchStep.decreasing = (searchStep.error < bestStep.error)

                alpha = self.__dampening(alpha, searchStep.decreasing)

                if searchStep.decreasing:
                    bestStep = searchStep
                else:
                    badSteps += 1
                    if badSteps > self.MAX_BAD:
                        if verbose:
                            log.warn(REGISTRATION_STOP)
                        break

                    # Restore the parameters from the previous best iteration.
                    p = bestStep.p.copy()

            # Computes the derivative of the error with respect to model
            # parameters.

            J = method.jacobian(warpedImage, template, tform, p)

            # Compute the parameter update vector.
            deltaP = self.__deltaP(J, e, alpha, p)

            # Evaluate stopping condition:
            if np.dot(deltaP.T, deltaP) < 1e-4:
                break

            # Update the estimated parameters.
            p = method.update(p, deltaP, tform)

        return bestStep, search

