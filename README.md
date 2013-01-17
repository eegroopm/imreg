Image Registration (imreg)
==========================

The "imreg" package implements fast image registration methods using Python, Cython and numerical tools (scipy, numpy). 

This package uses `semantic versioning <http://semver.org/>`.

References:
-----------

    S. Baker and I. Matthews. Equivalence and eﬃciency of image alignment algorithms. 
    In Proceedings of the 2001 IEEE Conference on Computer Vision and Pattern Recognition, 
    Volume 1, Pages 1090 – 1097, December 2001.

A very comprehensive 3 part series with Matlab implementations):

    http://www.ri.cmu.edu/research_project_detail.html?project_id=515&menu_id=261

Maintainers
-----------

   - Nathan Faggian
   - Riaan Van Den Dool
   - Stefan Van Der Walt

Testing
-------

[![Build Status](https://travis-ci.org/pyimreg/imreg.png?branch=master)](https://travis-ci.org/pyimreg/imreg)

Dependencies
------------

The required dependencies to build the software are:

  - python
  - numpy
  - scipy
  - cython 
  - py.test

Install
-------

This packages uses distutils, which is the default way of installing python modules. To install in your home directory, use:

    python setup.py install --home

To install for all users on Unix/Linux:

    python setup.py build
    sudo python setup.py install

Development
-----------

Follow: Fork + Pull Model::

    http://help.github.com/send-pull-requests/

