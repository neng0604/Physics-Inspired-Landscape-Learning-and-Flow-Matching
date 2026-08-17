import argparse
import os
import signal
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.data import PDBProtein
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def to_numpy_array(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def has_nan_or_inf(x):
    if isinstance(x, torch.Tensor):
        return torch.isnan(x).any().item() or torch.isinf(x).any().item()

    x = np.asarray(x)
    return np.isnan(x).any() or np.isinf(x).any()


def get_vina_affinity(vina_results, key):
    if vina_results is None:
        return None
    if key not in vina_results:
        return None
    try:
        return float(vina_results[key][0]['affinity'])
    except Exception:
        return None


class VinaTimeoutError(TimeoutError):
    pass


def _raise_vina_timeout(signum, frame):
    raise VinaTimeoutError("Vina call timed out")


def run_vina_with_timeout(vina_task, mode, exhaustiveness, timeout_seconds=0):
    timeout_seconds = int(timeout_seconds or 0)
    if timeout_seconds <= 0:
        return vina_task.run(mode=mode, exhaustiveness=exhaustiveness)
    old_handler = signal.signal(signal.SIGALRM, _raise_vina_timeout)
    signal.alarm(timeout_seconds)
    try:
        return vina_task.run(mode=mode, exhaustiveness=exhaustiveness)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def mean_pairwise_diversity(mols):
    fps = []
    for mol in mols:
        try:
            fps.append(Chem.RDKFingerprint(mol))
        except Exception:
            continue
    if len(fps) < 2:
        return None

    diversities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            diversities.append(1.0 - DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return float(np.mean(diversities)) if diversities else None


def load_reference_mol(ligand_root, ligand_filename):
    ligand_path = os.path.join(ligand_root, ligand_filename)
    mol = next(iter(Chem.SDMolSupplier(ligand_path)))
    if mol is None:
        raise ValueError(f'Failed to load reference ligand: {ligand_path}')
    return mol


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'], default='vina_dock')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--compute_reference_affinity', type=eval, default=False)
    parser.add_argument('--reference_metrics_path', type=str, default=None)
    parser.add_argument('--reference_ligand_root', type=str, default='./data/test_set')
    parser.add_argument('--vina_timeout_seconds', type=int, default=0)
    parser.add_argument('--success_qed_threshold', type=float, default=0.25)
    parser.add_argument('--success_sa_threshold', type=float, default=0.59)
    parser.add_argument('--success_vina_dock_threshold', type=float, default=-8.18)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')
    reference_metrics = None
    if args.reference_metrics_path:
        logger.info(f'Loading reference metrics cache: {args.reference_metrics_path}')
        reference_metrics = torch.load(args.reference_metrics_path, map_location='cpu')

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()
    reference_vina_cache = {}
    reference_chem_cache = {}
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        logger.info(f'Processing example {example_idx}: {r_name}')

        # ===== Step 1: load =====
        try:
            logger.info('[Step 1] torch.load start')
            r = torch.load(r_name)
            logger.info('[Step 1] torch.load done')
        except Exception as e:
            logger.warning(f'Failed to load {r_name}: {e}')
            continue
        # ===== Step 2: read traj =====
        try:
            logger.info('[Step 2] read pred_ligand_pos_traj / pred_ligand_v_traj start')
            all_pred_ligand_pos = r['pred_ligand_pos_traj']
            all_pred_ligand_v = r['pred_ligand_v_traj']
            num_samples += len(all_pred_ligand_pos)
            logger.info(f'[Step 2] traj loaded, num_samples_in_example = {len(all_pred_ligand_pos)}')
        except Exception as e:
            logger.exception(f'[ERROR] read traj failed: {r_name}')
            continue

        # ===== Step 3: iterate samples =====
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):

            logger.info(f'[Step 3] sample_idx={sample_idx} start')

            pred_pos, pred_v = pred_pos[args.eval_step], pred_v[args.eval_step]
            pred_pos = to_numpy_array(pred_pos)
            pred_v = to_numpy_array(pred_v)
            logger.info(f'[Step 3] sample_idx={sample_idx} eval_step done')

            if has_nan_or_inf(pred_pos):
                logger.warning(f'[Step 3] sample_idx={sample_idx} invalid positions: nan/inf found')
                continue
            if has_nan_or_inf(pred_v):
                logger.warning(f'[Step 3] sample_idx={sample_idx} invalid atom indices: nan/inf found')
                continue

            # stability check
            logger.info(f'[Step 3] sample_idx={sample_idx} atomic_number start')
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            logger.info(f'[Step 3] sample_idx={sample_idx} atomic_number done')

            all_atom_types += Counter(pred_atom_type)

            logger.info(f'[Step 3] sample_idx={sample_idx} stability start')
            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            logger.info(f'[Step 3] sample_idx={sample_idx} stability done')

            logger.info(f'[Step 3] sample_idx={sample_idx} pair_distance start')
            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist
            logger.info(f'[Step 3] sample_idx={sample_idx} pair_distance done')

            # reconstruction
            try:
                logger.info(f'[Step 3] sample_idx={sample_idx} aromatic start')
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                logger.info(f'[Step 3] sample_idx={sample_idx} aromatic done')

                logger.info(f'[Step 3] sample_idx={sample_idx} reconstruct start')
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
                logger.info(f'[Step 3] sample_idx={sample_idx} reconstruct done')

                logger.info(f'[Step 3] sample_idx={sample_idx} smiles start')
                smiles = Chem.MolToSmiles(mol)
                logger.info(f'[Step 3] sample_idx={sample_idx} smiles done: {smiles}')

                logger.info(f'[Step 3] sample_idx={sample_idx} mol stats: atoms={mol.GetNumAtoms()}, bonds={mol.GetNumBonds()}')
            except Exception as e:
                logger.exception(f'[Step 3] sample_idx={sample_idx} reconstruct failed: {e}')
                continue

            n_recon_success += 1

            if '.' in smiles:
                logger.warning(f'[Step 3] sample_idx={sample_idx} fragmented smiles, skip: {smiles}')
                continue
            n_complete += 1

            # ===== DEBUG 用：先跳過 docking，確認是不是 docking 炸 =====
            # logger.info(f'[Step 3] sample_idx={sample_idx} skip docking for debug')
            # continue

            # chemical and docking check
            try:
                logger.info(f'[Step 3] sample_idx={sample_idx} chem start')
                chem_results = scoring_func.get_chem(mol)
                logger.info(f'[Step 3] sample_idx={sample_idx} chem done')
                vina_results = None

                if args.docking_mode == 'qvina':
                    logger.info(f'[Step 3] sample_idx={sample_idx} qvina task create start')
                    vina_task = QVinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    logger.info(f'[Step 3] sample_idx={sample_idx} qvina task create done')

                    logger.info(f'[Step 3] sample_idx={sample_idx} qvina run start')
                    vina_results = vina_task.run_sync()
                    logger.info(f'[Step 3] sample_idx={sample_idx} qvina run done')

                    n_eval_success += 1

                elif args.docking_mode in ['vina_score', 'vina_dock']:
                    logger.info(f'[Step 3] sample_idx={sample_idx} vina task create start')
                    vina_task = VinaDockingTask.from_generated_mol(
                        mol, r['data'].ligand_filename, protein_root=args.protein_root)
                    logger.info(f'[Step 3] sample_idx={sample_idx} vina task create done')

                    logger.info(f'[Step 3] sample_idx={sample_idx} vina score_only start')
                    score_only_results = run_vina_with_timeout(
                        vina_task,
                        mode='score_only',
                        exhaustiveness=args.exhaustiveness,
                        timeout_seconds=args.vina_timeout_seconds,
                    )
                    logger.info(f'[Step 3] sample_idx={sample_idx} vina score_only done')

                    logger.info(f'[Step 3] sample_idx={sample_idx} vina minimize start')
                    minimize_results = run_vina_with_timeout(
                        vina_task,
                        mode='minimize',
                        exhaustiveness=args.exhaustiveness,
                        timeout_seconds=args.vina_timeout_seconds,
                    )
                    logger.info(f'[Step 3] sample_idx={sample_idx} vina minimize done')

                    vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }

                    if args.docking_mode == 'vina_dock':
                        logger.info(f'[Step 3] sample_idx={sample_idx} vina dock start')
                        docking_results = run_vina_with_timeout(
                            vina_task,
                            mode='dock',
                            exhaustiveness=args.exhaustiveness,
                            timeout_seconds=args.vina_timeout_seconds,
                        )
                        logger.info(f'[Step 3] sample_idx={sample_idx} vina dock done')
                        vina_results['dock'] = docking_results

                    n_eval_success += 1
                elif args.docking_mode == 'none':
                    n_eval_success += 1
            except Exception as e:
                logger.exception(f'[Step 3] sample_idx={sample_idx} evaluation failed: {e}')
                continue

            reference_vina = None
            reference_chem = None
            if reference_metrics is not None:
                ref_record = reference_metrics.get('records', {}).get(r['data'].ligand_filename)
                if ref_record is not None:
                    reference_vina = ref_record.get('vina')
                    reference_chem = ref_record.get('chem_results')
            elif args.compute_reference_affinity and args.docking_mode in ['vina_score', 'vina_dock']:
                ligand_filename = r['data'].ligand_filename
                if ligand_filename not in reference_vina_cache:
                    try:
                        logger.info(f'[Reference] vina task create start: {ligand_filename}')
                        ref_mol = load_reference_mol(args.reference_ligand_root, ligand_filename)
                        reference_chem_cache[ligand_filename] = scoring_func.get_chem(ref_mol)
                        ref_task = VinaDockingTask.from_original_data(
                            r['data'],
                            ligand_root=args.reference_ligand_root,
                            protein_root=args.protein_root,
                        )
                        ref_score_only = run_vina_with_timeout(
                            ref_task,
                            mode='score_only',
                            exhaustiveness=args.exhaustiveness,
                            timeout_seconds=args.vina_timeout_seconds,
                        )
                        ref_minimize = run_vina_with_timeout(
                            ref_task,
                            mode='minimize',
                            exhaustiveness=args.exhaustiveness,
                            timeout_seconds=args.vina_timeout_seconds,
                        )
                        ref_vina = {
                            'score_only': ref_score_only,
                            'minimize': ref_minimize,
                        }
                        if args.docking_mode == 'vina_dock':
                            ref_vina['dock'] = run_vina_with_timeout(
                                ref_task,
                                mode='dock',
                                exhaustiveness=args.exhaustiveness,
                                timeout_seconds=args.vina_timeout_seconds,
                            )
                        reference_vina_cache[ligand_filename] = ref_vina
                        logger.info(f'[Reference] vina done: {ligand_filename}')
                    except Exception as e:
                        logger.exception(f'[Reference] vina failed: {ligand_filename}: {e}')
                        reference_vina_cache[ligand_filename] = None
                        reference_chem_cache[ligand_filename] = None
                reference_vina = reference_vina_cache.get(ligand_filename)
                reference_chem = reference_chem_cache.get(ligand_filename)
            # # stability check
            # pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            # all_atom_types += Counter(pred_atom_type)
            # r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            # all_mol_stable += r_stable[0]
            # all_atom_stable += r_stable[1]
            # all_n_atom += r_stable[2]

            # pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            # all_pair_dist += pair_dist

            # # reconstruction
            # try:
            #     pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            #     mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            #     smiles = Chem.MolToSmiles(mol)
            # except Exception as e:
            #     if args.verbose:
            #         logger.warning('Reconstruct failed %s: %s' % (f'{example_idx}_{sample_idx}', str(e)))
            #     continue
            # n_recon_success += 1

            # if '.' in smiles:
            #     continue
            # n_complete += 1

            # # chemical and docking check
            # try:
            #     chem_results = scoring_func.get_chem(mol)
            #     if args.docking_mode == 'qvina':
            #         vina_task = QVinaDockingTask.from_generated_mol(
            #             mol, r['data'].ligand_filename, protein_root=args.protein_root)
            #         vina_results = vina_task.run_sync()
            #     elif args.docking_mode in ['vina_score', 'vina_dock']:
            #         vina_task = VinaDockingTask.from_generated_mol(
            #             mol, r['data'].ligand_filename, protein_root=args.protein_root)
            #         score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
            #         minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
            #         vina_results = {
            #             'score_only': score_only_results,
            #             'minimize': minimize_results
            #         }
            #         if args.docking_mode == 'vina_dock':
            #             docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
            #             vina_results['dock'] = docking_results
            #         else:
            #             vina_results = None

            #         n_eval_success += 1
            # except:
            #     if args.verbose:
            #         logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
            #     continue

            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            results.append({
                'mol': mol,
                'smiles': smiles,
                'source_result_file': os.path.basename(r_name),
                'source_data_id': int(os.path.basename(r_name)[:-3].split('_')[-1]),
                'source_example_idx': int(example_idx),
                'source_sample_idx': int(sample_idx),
                'subset_index': r.get('subset_index'),
                'dataset_index': r.get('dataset_index'),
                'ligand_filename': r['data'].ligand_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results,
                'reference_vina': reference_vina,
                'reference_chem': reference_chem,
            })
    logger.info(f'Evaluate done! {num_samples} samples in total.')

    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }
    print_dict(validity_dict, logger)

    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
        n_recon_success, n_complete, len(results)))

    if len(results) == 0:
        logger.warning('No molecules passed full evaluation; skipping bond, atom-type, chemistry, and docking summaries.')
        if args.save:
            torch.save({
                'stability': validity_dict,
                'bond_length': all_bond_dist,
                'all_results': results
            }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))
        sys.exit(0)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))

    grouped_mols = {}
    for r in results:
        grouped_mols.setdefault(r['ligand_filename'], []).append(r['mol'])
    diversity_by_target = [
        d for d in (mean_pairwise_diversity(mols) for mols in grouped_mols.values())
        if d is not None
    ]
    if diversity_by_target:
        logger.info('Diversity: Mean: %.3f Median: %.3f' % (
            np.mean(diversity_by_target), np.median(diversity_by_target)
        ))

    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

        if args.compute_reference_affinity or reference_metrics is not None:
            if reference_metrics is not None and reference_metrics.get('summary') is not None:
                ref_summary = reference_metrics['summary']
                logger.info('Reference QED: Mean: %.3f Median: %.3f' % (
                    ref_summary['qed']['mean'], ref_summary['qed']['median']
                ))
                logger.info('Reference SA : Mean: %.3f Median: %.3f' % (
                    ref_summary['sa']['mean'], ref_summary['sa']['median']
                ))
                logger.info('Reference Vina Score: Mean: %.3f Median: %.3f' % (
                    ref_summary['vina_score']['mean'], ref_summary['vina_score']['median']
                ))
                logger.info('Reference Vina Min: Mean: %.3f Median: %.3f' % (
                    ref_summary['vina_min']['mean'], ref_summary['vina_min']['median']
                ))
                if ref_summary.get('vina_dock') and ref_summary['vina_dock']['mean'] is not None:
                    logger.info('Reference Vina Dock: Mean: %.3f Median: %.3f' % (
                        ref_summary['vina_dock']['mean'], ref_summary['vina_dock']['median']
                    ))
            ref_chem_values = [
                r.get('reference_chem') for r in results
                if r.get('reference_chem') is not None
            ]
            if ref_chem_values:
                ref_qed = [r['qed'] for r in ref_chem_values]
                ref_sa = [r['sa'] for r in ref_chem_values]
                if reference_metrics is None:
                    logger.info('Reference QED: Mean: %.3f Median: %.3f' % (np.mean(ref_qed), np.median(ref_qed)))
                    logger.info('Reference SA : Mean: %.3f Median: %.3f' % (np.mean(ref_sa), np.median(ref_sa)))

            affinity_keys = [('score_only', 'Vina Score'), ('minimize', 'Vina Min')]
            if args.docking_mode == 'vina_dock':
                affinity_keys.append(('dock', 'Vina Dock'))
            for key, label in affinity_keys:
                ref_affinities = []
                comparisons = []
                for r in results:
                    gen_affinity = get_vina_affinity(r['vina'], key)
                    ref_affinity = get_vina_affinity(r.get('reference_vina'), key)
                    if gen_affinity is not None and ref_affinity is not None:
                        ref_affinities.append(ref_affinity)
                        comparisons.append(float(gen_affinity < ref_affinity))
                if ref_affinities:
                    if reference_metrics is None:
                        logger.info('Reference %s: Mean: %.3f Median: %.3f' % (
                            label, np.mean(ref_affinities), np.median(ref_affinities)
                        ))
                if comparisons:
                    logger.info('High Affinity (%s): Mean: %.3f' % (label, np.mean(comparisons)))

        if args.docking_mode == 'vina_dock':
            success_flags = [
                (
                    r['chem_results']['qed'] > args.success_qed_threshold and
                    r['chem_results']['sa'] > args.success_sa_threshold and
                    get_vina_affinity(r['vina'], 'dock') is not None and
                    get_vina_affinity(r['vina'], 'dock') < args.success_vina_dock_threshold
                )
                for r in results
            ]
            logger.info('Success Rate: Mean: %.3f' % np.mean(success_flags))

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.save:
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'diversity_by_target': diversity_by_target,
            'all_results': results
        }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))
