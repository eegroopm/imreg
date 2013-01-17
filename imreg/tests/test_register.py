import numpy as np

import scipy.ndimage as nd
import scipy.misc as misc

from imreg import model, register, sampler


def deform(image, p, deformationModel):
    """
    Warps an image.
    """

    coords = model.Coordinates([0, image.shape[0], 0, image.shape[1]])

    return sampler.bilinear(image, deformationModel(coords, p).tensor)


def pytest_generate_tests(metafunc):
    """
    Generates a set of test for the registration methods.
    """

    image = misc.lena()
    image = nd.zoom(image, 0.25)

    if metafunc.function is test_shift:

        for displacement in np.arange(-10., 10.):

            p = np.array([displacement, displacement])

            template = deform(image, p, model.Shift())

            metafunc.addcall(
                id='dx={}, dy={}'.format(p[0], p[1]),
                funcargs=dict(
                    image=image,
                    template=template,
                    p=p
                    )
                )

    if metafunc.function is test_affine:

        # test the displacement component
        for displacement in np.arange(-10., 10.):

            p = np.array([0., 0., 0., 0., displacement, displacement])

            template = deform(image, p, model.Affine())

            metafunc.addcall(
                    id='dx={}, dy={}'.format(p[4], p[5]),
                    funcargs=dict(
                        image=image,
                        template=template,
                        p=p
                        )
                    )


def test_shift(image, template, p):
    """
    Tests image registration using a shift deformation model.
    """

    shift = register.Register()

    # Coerce the image data into RegisterData.
    image = register.RegisterData(image)
    template = register.RegisterData(template)

    step, _search = shift.register(image, template, model.Shift())

    assert np.allclose(p, step.p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            step.p,
            p
            )


def test_affine(image, template, p):
    """
    Tests image registration using a affine deformation model.
    """

    affine = register.Register()

    # Coerce the image data into RegisterData.
    image = register.RegisterData(image)
    template = register.RegisterData(template)

    step, _search = affine.register(image, template, model.Affine())

    assert np.allclose(p, step.p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            step.p,
            p
            )
