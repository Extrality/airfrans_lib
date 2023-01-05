Dataset
=======

The AirfRANS dataset makes available numerical resolutions of the incompressible Reynolds-Averaged Navier-Stokes (RANS) equations over the NACA 4 and 5 digits series of airfoils and in a subsonic flight regime setup.

Its features are:

- 1000 simulations
- Reynolds number between 2 and 6 million
- Angle of attacks between -5° and 15°
- Airfoil drawn in the NACA 4 and 5 digit series
- Four machine learning tasks representing different challenges.
	
The four tasks are defined as followed:

- **Full data regime**: 800 simulations are used for the training and 200 are kept for testing. Both the trainset and the testset are drawn from the same distribution. This defines an interpolation task.
- **Scarce data regime**: Same testset as the `Full data regime` task but with only 200 simulations in the trainset. This also defines an interpolation task but in a low data regime scenario.
- **Reynolds extrapolation regime**: Simulations with Reynolds number between 3 and 5 million are kept for the trainset, the others are kept for the testset. This defines an extrapolation task for the reynolds number parameter.
- **Angle of attack extrapolation regime**: Simulations with angle of attack between -2.5° and 12.5° are kept for the trainset, the others are kept for the testset. This defines an extrapolation task for the angle of attack parameter.
	
Downloading the dataset
-----------------------
	
You can download the dataset by using the function :obj:`airfrans.dataset.download`

.. code-block:: python

	import airfrans as af
	
	af.dataset.download(root = PATH_TO_SAVING_DIRECTORY, file_name = 'Dataset', unzip = True, OpenFOAM = False)

for the pre-processed dataset where simulations have been cropped, minimum features have been kept and ``.vtu`` / ``.vtp`` files have been generated. For the raw OpenFOAM dataset, simply set ``True`` to the ``OpenFOAM`` argument. You can also directly download it `here <https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip>`_ for the pre-processed version and `there <https://data.isir.upmc.fr/extrality/NeurIPS_2022/OF_dataset.zip>`_ for the raw OpenFOAM.

Finally, you can access to a ready-to-use version via `PyTorch Geometric datasets <https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.AirfRANS>`_.

Loading the dataset
-------------------

After dowloading the pre-processed dataset, you can load the list of simulations in NumPy arrays with the function :obj:`airfrans.dataset.load`

.. code-block:: python

	dataset_list, dataset_name = af.dataset.load(root = PATH_TO_DATASET, task = 'full', train = True)

We recommend to handle this data with a Geometric Deep Learning library such as `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ or `Deep Graph Library <https://www.dgl.ai/>`_.

.. note::

	The dataset loaded when using :obj:`airfrans.dataset.Load` contains point clouds defined as the nodes of the simulation mesh. Do not hesitate to build a custom dataset if you want to use volume/surface sampling of simulations instead of the nodes of the mesh via the :class:`airfrans.Simulation`. The only constraint is that the test scores of models for the four target fields have to be computed at the position of the nodes of the simulation meshes.
