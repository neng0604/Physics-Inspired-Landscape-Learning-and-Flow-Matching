import os
import contextlib
import sys
import subprocess
import argparse
import json
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from openbabel import pybel
from meeko import MoleculePreparation
from meeko import obutils
from vina import Vina
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import AutoDockTools

from utils.reconstruct import reconstruct_from_generated
from utils.evaluation.docking_qvina import get_random_id, BaseDockingTask


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format == 'sdf':
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')

    def addH(self, polaronly=False, correctforph=True, PH=7):
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        sdf_block = self.ob_mol.write('sdf')
        rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
        AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
        self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
        obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None:
            preparator.write_pdbqt_file(lig_pdbqt)
            return
        else:
            return preparator.write_pdbqt_string()


class PrepProt(object):
    def __init__(self, pdb_file):
        self.prot = pdb_file

    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f:
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')]
            dry_lines = [l for l in lines if not 'HOH' in l]

        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file

    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


class VinaDock(object):
    def __init__(self, lig_pdbqt, prot_pdbqt):
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt

    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f:
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size

    def get_box(self, ref=None, buffer=0):
        '''
        ref: reference pdb to define pocket.
        buffer: buffer size to add

        if ref is not None:
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension
        else:
            use the entire protein to define pocket
        '''
        if ref is None:
            ref = self.prot_pdbqt
        self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print(self.pocket_center, self.box_size)

    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
        # The Vina Python API uses every visible CPU when cpu=0.  Batch jobs
        # can cap each independent docking worker through VINA_CPU to avoid
        # multiplying the full Slurm cpuset by the outer process count.
        if 'cpu' not in kwargs and os.environ.get('VINA_CPU'):
            kwargs['cpu'] = max(1, int(os.environ['VINA_CPU']))
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only':
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=1)
            score = v.energies(n_poses=1)[0][0]
        else:
            raise ValueError

        if not save_pose:
            return score
        else:
            if mode == 'score_only':
                pose = None
            elif mode == 'minimize':
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f:
                    v.write_pose(tmp.name, overwrite=True)
                with open(tmp.name, 'r') as f:
                    pose = f.read()

            elif mode == 'dock':
                pose = v.poses(n_poses=1)
            else:
                raise ValueError
            return score, pose


def _run_vina_impl(receptor_path, ligand_path, center, box_size, mode='dock', exhaustiveness=8, **kwargs):
    ligand_pdbqt = ligand_path[:-4] + '.pdbqt'
    protein_pqr = receptor_path[:-4] + '.pqr'
    protein_pdbqt = receptor_path[:-4] + '.pdbqt'

    lig = PrepLig(ligand_path, 'sdf')
    lig.get_pdbqt(ligand_pdbqt)

    prot = PrepProt(receptor_path)
    if not os.path.exists(protein_pqr):
        prot.addH(protein_pqr)
    if not os.path.exists(protein_pdbqt):
        prot.get_pdbqt(protein_pdbqt)

    dock = VinaDock(ligand_pdbqt, protein_pdbqt)
    dock.pocket_center = center
    dock.box_size = box_size
    score, pose = dock.dock(score_func='vina', mode=mode, exhaustiveness=exhaustiveness, save_pose=True, **kwargs)
    return [{'affinity': float(score), 'pose': pose}]


class VinaDockingTask(BaseDockingTask):

    @classmethod
    def from_generated_data(cls, data, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        ligand_rdmol = reconstruct_from_generated(data.clone())
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_original_data(cls, data, ligand_root='./data/crossdocked_pocket10', protein_root='./data/crossdocked',
                           **kwargs):
        protein_fn = os.path.join(
            os.path.dirname(data.ligand_filename),
            os.path.basename(data.ligand_filename)[:10] + '.pdb'
        )
        protein_path = os.path.join(protein_root, protein_fn)

        ligand_path = os.path.join(ligand_root, data.ligand_filename)
        ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
        return cls(protein_path, ligand_rdmol, **kwargs)

    @classmethod
    def from_generated_mol(cls, ligand_rdmol, ligand_filename, protein_root='./data/crossdocked', **kwargs):
        # load original pdb
        protein_fn = os.path.join(
            os.path.dirname(ligand_filename),
            os.path.basename(ligand_filename)[:10] + '.pdb'  # PDBId_Chain_rec.pdb
        )
        protein_path = os.path.join(protein_root, protein_fn)
        return cls(protein_path, ligand_rdmol, **kwargs)

    def __init__(self, protein_path, ligand_rdmol, tmp_dir='./tmp', center=None,
                 size_factor=1., buffer=5.0):
        super().__init__(protein_path, ligand_rdmol)
        # self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = protein_path
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        self.recon_ligand_mol = ligand_rdmol
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)

        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()
        self.ligand_rdmol = ligand_rdmol

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 20, 20, 20
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor + buffer

        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

    def run(self, mode='dock', exhaustiveness=8, **kwargs):
        seed = int(kwargs.pop('seed', 0))
        if kwargs:
            raise TypeError(f'Unsupported VinaDockingTask.run options: {sorted(kwargs)}')
        out_json = os.path.join(self.tmp_dir, f'{self.task_id}_{mode}.json')
        cmd = [
            sys.executable,
            os.path.realpath(__file__),
            '--worker',
            '--receptor_path', self.receptor_path,
            '--ligand_path', self.ligand_path,
            '--center_x', str(float(self.center[0])),
            '--center_y', str(float(self.center[1])),
            '--center_z', str(float(self.center[2])),
            '--size_x', str(float(self.size_x)),
            '--size_y', str(float(self.size_y)),
            '--size_z', str(float(self.size_z)),
            '--mode', mode,
            '--exhaustiveness', str(exhaustiveness),
            '--seed', str(seed),
            '--out_json', out_json,
        ]
        env = os.environ.copy()
        pythonpath = env.get('PYTHONPATH')
        env['PYTHONPATH'] = PROJECT_ROOT if not pythonpath else PROJECT_ROOT + os.pathsep + pythonpath
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=PROJECT_ROOT,
            env=env,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            stdout = proc.stdout.strip()
            details = stderr or stdout or f'worker exited with code {proc.returncode}'
            raise RuntimeError(f'Vina subprocess failed: {details}')
        if not os.path.exists(out_json):
            raise RuntimeError('Vina subprocess finished without output json')
        with open(out_json, 'r') as f:
            return json.load(f)


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--receptor_path', type=str, required=True)
    parser.add_argument('--ligand_path', type=str, required=True)
    parser.add_argument('--center_x', type=float, required=True)
    parser.add_argument('--center_y', type=float, required=True)
    parser.add_argument('--center_z', type=float, required=True)
    parser.add_argument('--size_x', type=float, required=True)
    parser.add_argument('--size_y', type=float, required=True)
    parser.add_argument('--size_z', type=float, required=True)
    parser.add_argument('--mode', type=str, choices=['score_only', 'minimize', 'dock'], required=True)
    parser.add_argument('--exhaustiveness', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_json', type=str, required=True)
    args = parser.parse_args()

    results = _run_vina_impl(
        receptor_path=args.receptor_path,
        ligand_path=args.ligand_path,
        center=[args.center_x, args.center_y, args.center_z],
        box_size=[args.size_x, args.size_y, args.size_z],
        mode=args.mode,
        exhaustiveness=args.exhaustiveness,
        seed=args.seed,
    )
    with open(args.out_json, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    _worker_main()
