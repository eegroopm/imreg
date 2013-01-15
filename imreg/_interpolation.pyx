#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp

from libc.math cimport ceil, floor

def nearest(
    cnp.ndarray[double, ndim=3,  mode="c"] warp,
    cnp.ndarray[double, ndim=2,  mode="c"] image,
    cnp.ndarray[double, ndim=2,  mode="c"] output
    ):
    """ Computes a nearest neighbor sample """

    cdef int out_rows = warp[0].shape[1]
    cdef int out_cols = warp[0].shape[0]

    cdef int img_rows = image.shape[1]
    cdef int img_cols = image.shape[0]

    cdef double rhat, chat
    cdef int r, c

    for r in xrange(out_rows):
       for c in xrange(out_cols):
            output[r][c] = nearest_neighbour_interpolation(
                                <double*> image.data,
                                img_rows,
                                img_cols,
                                warp[0, r, c],
                                warp[1, r, c],
                                'C',
                                0.0
                                )
    return 0


def bilinear(
    cnp.ndarray[double, ndim=3,  mode="c"] warp,
    cnp.ndarray[double, ndim=2,  mode="c"] image,
    cnp.ndarray[double, ndim=2,  mode="c"] output
    ):
    """ Computes a bilinear sample """

    cdef int out_rows = warp[0].shape[1]
    cdef int out_cols = warp[0].shape[0]

    cdef int img_rows = image.shape[1]
    cdef int img_cols = image.shape[0]

    cdef double rhat, chat
    cdef int r, c


    for r in xrange(out_rows):
       for c in xrange(out_cols):
            output[r][c] = bilinear_interpolation(
                                <double*> image.data,
                                img_rows,
                                img_cols,
                                warp[0, r, c],
                                warp[1, r, c],
                                'C',
                                0.0
                                )
    return 0


# Code below is directly from skimage -
#  - latest commit @ 4ff97da8b15c987b5eb5b2944395cff812715e0e


cdef inline int round(double r):
    return <int>((r + 0.5) if (r > 0.0) else (r - 0.5))


cdef inline double nearest_neighbour_interpolation(double* image, int rows,
                                                   int cols, double r,
                                                   double c, char mode,
                                                   double cval):
    """Nearest neighbour interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

    """

    return get_pixel2d(image, rows, cols, <int>round(r), <int>round(c),
                       mode, cval)


cdef inline double bilinear_interpolation(double* image, int rows, int cols,
                                          double r, double c, char mode,
                                          double cval):
    """Bilinear interpolation at a given position in the image.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : double
        Position at which to interpolate.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Interpolated value.

    """
    cdef double dr, dc
    cdef int minr, minc, maxr, maxc

    minr = <int>floor(r)
    minc = <int>floor(c)
    maxr = <int>ceil(r)
    maxc = <int>ceil(c)
    dr = r - minr
    dc = c - minc
    top = (1 - dc) * get_pixel2d(image, rows, cols, minr, minc, mode, cval) \
          + dc * get_pixel2d(image, rows, cols, minr, maxc, mode, cval)
    bottom = (1 - dc) * get_pixel2d(image, rows, cols, maxr, minc, mode, cval) \
             + dc * get_pixel2d(image, rows, cols, maxr, maxc, mode, cval)
    return (1 - dr) * top + dr * bottom


cdef inline double get_pixel2d(double* image, int rows, int cols, int r, int c,
                               char mode, double cval):
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : double array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'R', 'N'}
        Wrapping mode. Constant, Wrap, Reflect or Nearest.
    cval : double
        Constant value to use for constant mode.

    Returns
    -------
    value : double
        Pixel value at given position.

    """
    if mode == 'C':
        if (r < 0) or (r > rows - 1) or (c < 0) or (c > cols - 1):
            return cval
        else:
            return image[r * cols + c]
    else:
        return image[coord_map(rows, r, mode) * cols + coord_map(cols, c, mode)]


cdef inline int coord_map(int dim, int coord, char mode):
    """
    Wrap a coordinate, according to a given mode.

    Parameters
    ----------
    dim : int
        Maximum coordinate.
    coord : int
        Coord provided by user.  May be < 0 or > dim.
    mode : {'W', 'R', 'N'}
        Whether to wrap or reflect the coordinate if it
        falls outside [0, dim).

    """
    dim = dim - 1
    if mode == 'R': # reflect
        if coord < 0:
            # How many times times does the coordinate wrap?
            if <int>(-coord / dim) % 2 != 0:
                return dim - <int>(-coord % dim)
            else:
                return <int>(-coord % dim)
        elif coord > dim:
            if <int>(coord / dim) % 2 != 0:
                return <int>(dim - (coord % dim))
            else:
                return <int>(coord % dim)
    elif mode == 'W': # wrap
        if coord < 0:
            return <int>(dim - (-coord % dim))
        elif coord > dim:
            return <int>(coord % dim)
    elif mode == 'N': # nearest
        if coord < 0:
            return 0
        elif coord > dim:
            return dim

    return coord
