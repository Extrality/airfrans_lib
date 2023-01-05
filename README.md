# AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions

The AirfRANS dataset makes available numerical resolutions of the incompressible Reynolds-Averaged Navier–Stokes (RANS) equations over the NACA 4 and 5 digits series of airfoils and in a subsonic flight regime setup. Readthedocs documentation is available [here](https://airfrans.readthedocs.io/en/latest/index.html).

## Features
- Access to 1000 simulations.
- Reynolds number between 2 and 6 million.
- Angle of attack between -5° and 15°.
- Airfoil drawn in the NACA 4 and 5 digits series.
- Four machine learning tasks representing different challenges.

## Installation
Install with
```
pip install airfrans
```

## Usage
### Downloading the dataset
From python:
```
import airfrans as af

af.dataset.download(root = PATH_TO_SAVING_DIRECTORY, unzip = True)
```
You can also directly download a ready-to-use version of the dataset in the [PyTorch Geometric library](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.AirfRANS)
Finally, you can directly download the dataset in the raw OpenFOAM version [here](https://data.isir.upmc.fr/extrality/NeurIPS_2022/OF_dataset.zip), or in the more friendly pre-processed version [here](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip).

### Loading the dataset
From python:
```
import airfrans as af

dataset, dataname = af.dataset.load(root = PATH_TO_DATASET, task = TASK, train = True)
```
The tasks are the one presented in the [associated paper](https://arxiv.org/pdf/2212.07564.pdf). You can choose between `'full'`, `'scarce'`, `'reynolds`' and `'aoa'`.
The dataset loaded in this case is the same as the one you can directly access via the [PyTorch Geometric library](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.AirfRANS). If you want more flexibility about the sampling of each simulation for the inputs or targets, please feel free to build a custom loader with the help of the `'Simulation'` class presented in the following. We highly recommend to handle those data with a Gemetric Deep Learning library such as [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) or [Deep Graph Library](https://www.dgl.ai/).

### Simulation class
The `'Simulation'` class is an object to facilitate the manipulation of AirfRANS simulations. Given the root folder of where the directories of the simulations have been saved and the name of a simulation you can easily manipulate it.
```
import airfrans as af

name = 'airFoil2D_SST_57.872_7.314_5.454_3.799_13.179'
simulation = af.Simulation(root = PATH_TO_DATASET, name = name)
```
See the documentation for more details about this object.

## License
This project is licensed under the [ODbL-1.0 License](https://opendatacommons.org/licenses/odbl/1-0/).

## Reference
The original paper accepted at the 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks can be found [here](https://openreview.net/forum?id=Zp8YmiQ_bDC) and the preprint [here](https://arxiv.org/abs/2212.07564). Please cite this paper if you use this dataset in your own work.
```
@inproceedings{
bonnet2022airfrans,
title={Airf{RANS}: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier{\textendash}Stokes Solutions},
author={Florent Bonnet and Jocelyn Ahmed Mazari and Paola Cinnella and Patrick Gallinari},
booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
year={2022},
url={https://openreview.net/forum?id=Zp8YmiQ_bDC}
  }
```
