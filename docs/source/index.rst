:github_url: https://github.com/Extrality/airfrans_lib

AirfRANS documentation
======================

**AirfRANS** is a Python library for handling simulations coming from the AirfRANS dataset which makes available numerical resolutions of the incompressible Reynolds-Averaged Navier–Stokes (RANS) equations over the NACA 4 and 5 digits series of airfoils and in a subsonic flight regime setup.

It consists of a utility for downloading and loading the dataset under a NumPy format and a class to manipulate and execute some basic operations on the simulations. In particular, it allows to compute the force coefficent straightforwardly and it includes a method to visualize boundary layers and trails. Moreover, it allows to sample uniformly or with respect to the mesh density and to get rid of the mesh points constraint for training.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/introduction
   notes/installation
   notes/dataset
   notes/simulation
   notes/visualization

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package Reference

   modules/root
   modules/dataset
   modules/sampling
   modules/naca
