# Building from Source

## Python/PIP

Prepare the source distribution

> python setup.py sdist

To build a wheel you need to install wheel into your Python build environment

> pip install wheel

fsl_sub is only compatible with python 3 so you will be building a Pure Python Wheel

> python setup.py bdist_wheel

## Conda Build

To build a conda package:

> cd condaRecipies
> conda-build ..
