About
=====

*imreg*, short for image registration, is a python package for image registration
built ontop of scipy and numpy.

It is currently maintained by:
   - Nathan Faggian,
   - Riaan Van Den Dool,
   - Stefan Van Der Walt.

Testing
=======

[![Build Status](https://travis-ci.org/pyimreg/imreg.png?branch=master)](https://travis-ci.org/pyimreg/imreg)

Dependencies
============

The required dependencies to build the software are:

  - python >= 2.5
  - numpy >= 1.5
  - scipy >= 0.9
  - py.test >= 2.0

Install
=======

This packages uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

    python setup.py install --home

To install for all users on Unix/Linux::

    python setup.py build
    sudo python setup.py install

Development
===========

Follow: Fork + Pull Model::

    http://help.github.com/send-pull-requests/

