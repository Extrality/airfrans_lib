
import json
from urllib.request import urlretrieve
import os
import os.path as osp
import zipfile

import numpy as np
from tqdm import tqdm

from airfrans.simulation import Simulation

class DownloadProgressBar(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b = 1, bsize = 1, tsize = None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b*bsize - self.n) # also sets self.n = b * bsize

def download(root, file_name = 'Dataset', unzip = True, OpenFOAM = False):
    """
    Download AirfRANS dataset.

    Args:
        root (str): Root directory where the dataset will be downloaded and unzipped.
        file_name (str, optional): Name of the dataset file. Default: ``'Dataset'``   
        unzip (bool, optional): If ``True``, unzip the dataset file. Default: ``True``
        OpenFOAM (bool, optional): If ``True``, it will download the raw OpenFOAM simulation
            with no post-processing to manipulate it through PyVista. If ``False``, it will 
            download the ``.vtu`` and ``.vtp`` of cropped simulations with a reduced quantity
            of features. Those cropped simulations have been used to train models proposed
            in the associated paper. Default: ``False``
    """
    if OpenFOAM:
        url = 'https://data.isir.upmc.fr/extrality/NeurIPS_2022/OF_dataset.zip'
    else:
        url = 'https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip'

    os.makedirs(root, exist_ok = True)
    with DownloadProgressBar(unit = 'B', unit_scale = True, miniters = 1, unit_divisor = 1024, desc = 'Downloading AirfRANS') as t:
        urlretrieve(url, filename = osp.join(root, file_name + '.zip'), reporthook = t.update_to)

    if unzip:
        print("Extracting " + file_name + ".zip at " + root + "...")
        with zipfile.ZipFile(osp.join(root, file_name + '.zip'), 'r') as zipf:
            zipf.extractall(root)

def load(root, task, train = True):
    """
    The different tasks (``'full'``, ``'scarce'``, ``'reynolds'``,
    ``'aoa'``) define the utilized training and test splits. Please note
    that the test set for the ``'full'`` and ``'scarce'`` tasks are the same.
    Each simulation is given as a point cloud defined via the nodes of the
    simulation mesh. Each point of a point cloud is described via 7
    features: its position (in meters), the inlet velocity (two components in meter per second), the
    distance to the airfoil (one component in meter), and the normals (two
    components in meter, set to 0 if the point is not on the airfoil).

    Each point is given a target of 4 components for the underlying regression
    task: the velocity (two components in meter per second), the pressure
    divided by the specific mass (one component in meter squared per second
    squared), the turbulent kinematic viscosity (one component in meter squared
    per second).

    Finally, a boolean is attached to each point to inform if this point lies on
    the airfoil or not.

    The output is a tuple of a list of np.ndarray of shape `(N, 7 + 4 + 1)`, where N is the
    number of points in each simulation and where the features are ordered as presented
    in this documentation, and a list of name for the each corresponding simulation.

    We highly recommend to handle those data with the help of a Geometric Deep
    Learning library such as PyTorch Geometric or Deep Graph Library.

    Args:
        root (string): Root directory where the simulation directories have been saved.
        task (string): The task to study (``'full'``, ``'scarce'``, ``'reynolds'``, ``'aoa'``) 
            that defines the utilized training and test splits.
        train (bool, optional): If ``True``, loads the training dataset, otherwise the 
            test dataset. Default: ``True``
    """
    tasks = ['full', 'scarce', 'reynolds', 'aoa']
    if task not in tasks:
        raise ValueError(f"Expected 'task' to be in {tasks} "
                            f"got '{task}'")

    taskk = 'full' if task == 'scarce' and not train else task
    split = 'train' if train else 'test'

    with open(osp.join(root, 'manifest.json'), 'r') as f:
        manifest = json.load(f)[taskk + '_' + split]

    data_list = []
    name_list = []
    for s in tqdm(manifest, desc = f'Loading dataset (task: {taskk}, split: {split})'):
        simulation = Simulation(root = root, name = s)
        inlet_velocity = (np.array([np.cos(simulation.angle_of_attack),\
                np.sin(simulation.angle_of_attack)])*simulation.inlet_velocity).reshape(1, 2)\
                *np.ones_like(simulation.sdf)

        attribute = np.concatenate([
            simulation.position,
            inlet_velocity,
            simulation.sdf,
            simulation.normals,
            simulation.velocity,
            simulation.pressure,
            simulation.nu_t,
            simulation.surface.reshape(-1, 1)
        ], axis = -1)

        data_list.append(attribute)
        name_list.append(s)
    
    return data_list, name_list
