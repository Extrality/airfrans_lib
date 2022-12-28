# AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions

The AirfRANS dataset makes available numerical resolutions of the incompressible Reynolds-Averaged Navier–Stokes (RANS) equations over the NACA 4 and 5 digits series of airfoils and in a subsonic flight regime setup.

## Features
- Access to 1000 simulations.
- Reynolds number between 2 and 6 million.
- Angle of attack between -5° and 15°.
- Airfoil drawn in the NACA 4 and 5 digits series.
- Four machine learning tasks representing different challenges.

## Installation
Install with
`pip install airfrans`

## Usage
# Downloading the dataset
From python:
`
  import airfrans as af
  af.download.Download(root = PATH_TO_DATASET, unzip = True)
`
You can also directly download a ready-to-use version of the dataset in the [PyTorch Geometric library](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.AirfRANS)
Finally, you can directly download the dataset in the raw OpenFOAM version [here](https://data.isir.upmc.fr/extrality/NeurIPS_2022/OF_dataset.zip), or in the more friendly pre-processed version [here](https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip).

# Loading the dataset

## License
This project is licensed under the [MIT license](https://github.com/Extrality/airfrans_lib/blob/main/LICENSE)

## Reference
The original paper accepted at the 36th Conference on Neural Information Processing Systems (NeurIPS 2022) Track on Datasets and Benchmarks can be found [here](https://openreview.net/forum?id=Zp8YmiQ_bDC) and the preprint [here](https://arxiv.org/abs/2212.07564). Please cite this paper if you use this dataset in your own work.
`
  @inproceedings{
  bonnet2022airfrans,
  title={Airf{RANS}: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier{\textendash}Stokes Solutions},
  author={Florent Bonnet and Jocelyn Ahmed Mazari and Paola Cinnella and Patrick Gallinari},
  booktitle={Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2022},
  url={https://openreview.net/forum?id=Zp8YmiQ_bDC}
  }
`
