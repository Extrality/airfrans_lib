Introduction
============

The AirfRANS dataset is a collection of numerical simulations solving the incompressible Reynolds-Averaged Navier-Stokes equations over two dimensional airfoils in a subsonic flight regime. The associated `paper <https://openreview.net/forum?id=Zp8YmiQ_bDC>`_ has been accepted at the 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks. In addition to this library two GitHub repositories have been proposed to reconduct the paper `experiments <https://github.com/Extrality/AirfRANS>`_ and to generate new `compressible or incompressible simulations over NACA airfoils <https://github.com/Extrality/NACA_simulation>`_. The setup to generate those simulations have been confronted to the Langley Research Center experiments available in the `Turbulence Modeling Resource <https://turbmodels.larc.nasa.gov/>`_ for the NACA 0012 and 4412.

This dataset has been built to lower the potential barrier between Machine Learning and Physics research communities. It proposes data on a simple but realistic case which already includes some of the major challenges of Machine Learning for solving Fluid Dynamics, namely:

- working with unstructured data coming from raw numerical simulations,
- being able to deal with the number of nodes required in simulation meshes (from hundreds of thousands to hundreds of million in 3D cases),
- treating cases with a realistic Reynolds number,
- regressing the entire velocity, pressure and turbulent viscosity fields from a geometry and the boundary conditions,
- being accurate on global forces or coefficient such as drag and lift,
- being consistent between the predicted fields and the predicted forces,
- regressing accurately boundary layers and area of simulations where sharp signals appear,
- producing solutions that respect the conservation equation and the momentum equations.

We hope that this library will ease the manipulation of such simulations and the usage of the AirfRANS dataset.

Raw OpenFOAM data
-----------------

The dataset comes under different form. A pre-processed version of cropped simulations and including only the minimum number of fields is proposed as a work basis. A full raw OpenFOAM data version is also available but its manipulation necessitates some basic knowledge of how OpenFOAM works. However, raw data include more information than the processed data and especially if you are interested in the gradient of the fields. Each raw data contains each term of the momentum and conservation equations as fields that could be used, for example, to compare the gradients of the approximation with the gradients of the simulation.

Tutorials
---------

We used `OpenFOAM v2112 <https://www.openfoam.com/>`_ to generate our simulation. The manipulation and visualization of the results can be done with `ParaView <https://www.paraview.org/>`_ and/or with a pythonic interface such as `PyVista <https://docs.pyvista.org/>`_. Finally, the treatment of those data in a deep learning point of view can be done with Geometric Deep Learning library such as `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/#>`_ or `Deep Graph Library <https://www.dgl.ai/>`_. As those tools and domains are not necessarily well known in the Machine Learning community, we would like to share some tutorials and books that helped us to be more comfortable with the subject:

- One of the `OpenFOAM wiki <https://wiki.openfoam.com/Main_Page>`_ is a must for learning this powerful tool. For just a taste, you can follow the `First Glimpse Series <https://wiki.openfoam.com/%22first_glimpse%22_series>`_ and for a more in-depth introduction, the `Three Weeks Series <https://wiki.openfoam.com/index.php?title=%223_weeks%22_series>`_.
- Concerning ParaView, a part of the Three Weeks Series is dedicated to it but can be followed independently. You can find it `here <https://wiki.openfoam.com/index.php?title=Visualization_by_Joel_Guerrero>`_.
- This `book <https://holzmann-cfd.com/community/publications>`_ proposes an overview of the mathematics in OpenFOAM.
- The `Turbulence Modeling for CFD <https://cfd.spbstu.ru/agarbaruk/doc/2006_Wilcox_Turbulence-modeling-for-CFD.pdf>`_ book for understanding how to model turbulence in CFD.
- The `Fundamentals of Aerodynamics <https://aviationdose.com/wp-content/uploads/2020/01/Fundamentals-of-aerodynamics-6-Edition.pdf>`_ book for an aerodynamics centered presentation of fluid dynamics.
- More fundamentaly, the `Fluid Mechanics <https://phys.au.dk/~srf/hydro/Landau+Lifschitz.pdf>`_ book for a general introduction to fluid mechanics.

License
-------

This dataset is under the `Open Data Commons Open Database License (ODbL) v1.0 <https://opendatacommons.org/licenses/odbl/1-0/>`_ and this library is under the `MIT License <https://github.com/Extrality/airfrans_lib/blob/main/LICENSE>`_.

This work is proposed by `Extrality <https://www.extrality.ai/>`_ and the `MLIA <https://www.isir.upmc.fr/equipes/mlia/>`_ team of Sorbonne Université, Paris.
