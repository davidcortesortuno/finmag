
[![CircleCI](https://circleci.com/gh/fangohr/finmag.svg?style=svg&circle-token=6e89ca6e2d8bb3dadd4ac9ec84bec71d91336f9c)](https://circleci.com/gh/fangohr/finmag)

FinMag
======

- a thin (and mostly) Python layer (hopefully!) on top of
FEniCS/Dolfin to enable Python-scripted multi-physics micromagnetic
simulations.

- finmag solves micromagnetic problems using finite elements

The GitHub page of the project is https://github.com/fangohr/finmag

The code is developed by Anders Johansen, Dmitri Chernyshenko, Gabriel Balaban, Hans Fangohr,	Marc-Antonio Bisotti, Maximilian Albert, Weiwei Wang, Marijan Beg, Mark Vousden, and Beckie Carey at the University of Southampton.

This is an early version; contributions and pull requests to both the code and documentation are welcome.

Documentation
-------------
The documentation is available in the form of Jupyter notebooks is available in `doc/ipython_notebooks_src` directory.

Installation
------------
Finmag dependencies can be installed by running an appropriate script (for the Ubuntu version you use) in `install/` directory or by making a Docker container with Dockerfile in `install/docker/`.

If you decide to install dependencies using a shell script in `install/` directory, you will need to add the path to the `finmag/src/` directory to your $PYTHONPATH.

# How to cite Finmag

?

# Acknowledgement 

We acknowledge financial support from

- EPSRC’s Doctoral Training Centre in Complex System Simulation
  (EP/G03690X/1), http://icss.soton.ac.uk
