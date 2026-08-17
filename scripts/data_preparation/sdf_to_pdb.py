import argparse
import pickle
from tqdm import tqdm
from rdkit import Chem
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ligand_sdf_path', type=str, default='./example/2z3h_ligand.sdf')
    args = parser.parse_args()

    sdf_supplier = Chem.SDMolSupplier(args.ligand_sdf_path)
    for mol in sdf_supplier:
        if mol is None:
              print('Skipping %s' % args.ligand_sdf_path)
              continue

        # 将分子写入 PDB 文件
        ligand_pdb_path = args.ligand_sdf_path.replace('.sdf', '.pdb')
        with open(ligand_pdb_path, 'w') as pdb_file:
            pdb_file.write(Chem.MolToPDBBlock(mol))
