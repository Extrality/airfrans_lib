from urllib.request import urlretrieve
from tqdm import tqdm
import os.path as osp
import zipfile

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

def Download(root, file_name = 'Dataset', unzip = True, OpenFOAM = False):
    """
    Download AirfRANS dataset.

    Args:
        root (str): Root directory where the dataset will be downloaded and unzipped.
        file_name (str, optional): Name of the dataset file. Default ``'Dataset'``
        unzip (bool, optional): If ``True``, unzip the dataset file. Default ``True``
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

    with DownloadProgressBar(unit = 'B', unit_scale = True, miniters = 1, unit_divisor = 1024, desc = 'Downloading AirfRANS') as t:
        urlretrieve(url, filename = osp.join(root, file_name + '.zip'), reporthook = t.update_to)

    print('Extracting ' + file_name + '.zip at ' + root + '...')
    if unzip:
        with zipfile.ZipFile(osp.join(root, file_name + '.zip'), 'r') as zipf:
            zipf.extractall(root)