import torch
from torch.utils.data import Subset
from datasets.pl_pair_dataset import PocketLigandPairDataset
from utils.pdbbind import PDBBindDataset


def get_dataset_pdbbind(config, transform, emb_path, heavy_only):
    name = config.name
    root = config.path
    if name == 'pdbbind':
        dataset = PDBBindDataset(root, transform, emb_path, heavy_only)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split = torch.load(config.split)
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
