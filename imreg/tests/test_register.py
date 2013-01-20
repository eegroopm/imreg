import numpy as np

import scipy.ndimage as nd
import scipy.misc as misc

from imreg import model, register, sampler, metric


def deform(image, p, deformationModel):
    """
    Warps an image.
    """

    coords = model.Coordinates([0, image.shape[0], 0, image.shape[1]])

    return sampler.bilinear(image, deformationModel(p, coords).tensor)


def pytest_generate_tests(metafunc):
    """
    Generates a set of test for the registration methods.
    """
    methods = [
        ('additive', metric.forwardsAdditive),
        ('compositional', metric.forwardsCompositional),
        ('inverse-compositional', metric.inverseCompositional)
        ]

    image = misc.lena()
    image = nd.zoom(image, 0.25)

    if metafunc.function is test_shift:

        for (method_name, method) in methods:

            for displacement in np.arange(-10., 11., 2.):
                p = np.array([displacement, displacement])
                template = deform(image, p, model.Shift())

                metafunc.addcall(
                    id='method={}, dx={}, dy={}'.format(method_name, p[0], p[1]),
                    funcargs=dict(
                        image=image,
                        template=template,
                        method=method,
                        p=p
                        )
                    )

    if metafunc.function is test_affine:

        for (method_name, method) in methods:
            # test the displacement component
            for displacement in np.arange(-10., 10., 2.):
                p = np.array([0., 0., 0., 0., displacement, displacement])
                template = deform(image, p, model.Affine())
                metafunc.addcall(
                        id='method={}, dx={}, dy={}'.format(method_name, p[4], p[5]),
                        funcargs=dict(
                            image=image,
                            template=template,
                            method=method,
                            p=p
                            )
                        )


def test_shift(image, template, method, p):
    """
    Tests image registration using a shift deformation model.
    """

    shift = register.Register()

    # Coerce the image data into RegisterData.
    image = register.RegisterData(image)
    template = register.RegisterData(template)

    step, _search = shift.register(image, template, model.Shift(), method=method)

    assert np.allclose(p, step.p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            step.p,
            p
            )


def test_affine(image, template, method, p):
    """
    Tests image registration using a affine deformation model.
    """

    affine = register.Register()

    # Coerce the image data into RegisterData.
    image = register.RegisterData(image)
    template = register.RegisterData(template)

    step, _search = affine.register(image, template, model.Affine(), method=method)

    assert np.allclose(p, step.p, atol=0.5), \
        "Estimated p: {} not equal to p: {}".format(
            step.p,
            p
            )
