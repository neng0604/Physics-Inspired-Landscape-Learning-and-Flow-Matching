import argparse
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyKVFinder import *
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pocket_path', type=str, default='./example/2z3h_pocket10.pdb')
    parser.add_argument('--ligand_path', type=str, default='./example/2z3h_ligand.pdb')
    args = parser.parse_args()

    atomic = read_pdb(args.pocket_path)
    vertices = get_vertices(atomic)
    latomic = read_pdb(args.ligand_path)
    ncav, cavities = detect(atomic, vertices, latomic=latomic, ligand_cutoff=10.0)
    surface, volume, area = spatial(cavities)
    print('Volume:', float(volume['KAA']))
    print('Area:', float(area['KAA']))
