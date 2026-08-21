# LandFlow

LandFlow adds a lightweight learned route potential to a frozen PAFlow sampler.
The potential is trained on future-quality targets accumulated along generated
trajectories. During sampling, its negative coordinate gradient is normalized
and added as a scale-controlled residual to the frozen PAFlow coordinate
velocity.

The released route potential has 115,842 trainable parameters (0.116M). PAFlow
and its affinity predictor remain frozen during LandFlow training.

## Repository layout

```text
configs/       portable train/test sampling configurations
models/        PAFlow and LandFlow model definitions
scripts/       sampling, bank construction, training, and evaluation
workflows/     ordered command-line entry points
huggingface/   model and dataset cards
artifacts.json expected artifact sizes and SHA-256 checksums
```

Large checkpoints, the trajectory bank, and the PAFlow-ready CrossDocked
pocket10 data are hosted separately on Hugging Face.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate landflow
```

The exact environment snapshot from the development machine is retained as
`environment.lock.yml`. It is useful for debugging but less portable than
`environment.yml`.

## 2. Download released artifacts

```bash
export HF_MODEL_REPO=YOUR_HF_USERNAME/LandFlow
export HF_DATA_REPO=YOUR_HF_USERNAME/LandFlow-data
python scripts/download_artifacts.py --include-crossdocked --extract-test-set
```

This restores the following files:

```text
pretrained_models/pretrained_flow.pt
pretrained_models/atom_num_model.pt
checkpoints/landflow.pt
artifacts/trajectory_bank.pt
data/crossdocked_pocket10_pose_split.pt
data/atom_num_dataset.pkl
data/reference_metrics.pt
data/crossdocked_pocket10_with_protein.tar.gz
data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
data/test_set.tar.gz
data/test_set/
```

The 3.4 GB pocket source archive is retained for preprocessing reproducibility
and expands to about 15 GB. It is not extracted by default because sampling and
evaluation use the LMDB and `data/test_set/` directly.

## 3. Verify the installation

The smoke test loads the released checkpoint, verifies all available hashes,
counts the 115,842 parameters, and checks that the potential produces a finite,
nonzero ligand-coordinate gradient.

```bash
python scripts/verify_install.py --require-crossdocked
```

## 4. Sample one test pocket

Run this first before submitting the full experiment:

```bash
DATA_START=0 DATA_END=0 NUM_SAMPLES=2 bash workflows/04_sample_landflow.sh
DOCKING_MODE=none bash workflows/05_evaluate.sh
```

For the paper setting (100 pockets, 10 ligands per pocket, `rho=0.30`):

```bash
DATA_START=0 DATA_END=99 NUM_SAMPLES=10 bash workflows/04_sample_landflow.sh
DOCKING_MODE=vina_dock EXHAUSTIVENESS=16 bash workflows/05_evaluate.sh
```

Each `workflows/*.sh` file can run directly on a GPU compute node or be passed
to the local scheduler, for example:

```bash
sbatch -A YOUR_ACCOUNT -p YOUR_GPU_PARTITION --gres=gpu:1 \
  --cpus-per-task=8 --mem=64G --time=04:00:00 \
  workflows/04_sample_landflow.sh
```

## 5. Retrain the route potential

The fastest exact retraining path starts from the released trajectory bank:

```bash
bash workflows/03_train_potential.sh
```

Before a full retraining run, use the short training smoke test:

```bash
EPOCHS=1 MAX_TRAIN_BATCHES=2 VAL_MAX_BATCHES=2 \
  OUTPUT_DIR=outputs/training_smoke bash workflows/03_train_potential.sh
```

To rebuild the bank from frozen PAFlow trajectories:

```bash
GAMMAS="0 150 350" DATA_START=0 DATA_END=99 NUM_SAMPLES=10 \
  bash workflows/01_collect_trajectories.sh
bash workflows/02_build_bank.sh
bash workflows/03_train_potential.sh
```

The three trajectory settings create 3,000 trajectories. With 50 recorded
states per trajectory, the bank contains 150,000 supervised state examples.

## Reproducibility boundary

`rho=0` means the frozen PAFlow path. The main LandFlow setting uses the same
PAFlow prior-affinity strength (`pos_grad_w=250`) plus the calibrated route
residual (`rho=0.30`, no projection, activation midpoint 0.50). Test pockets are
not used to construct the trajectory bank or train the route potential.

See [docs/HANDOFF.md](docs/HANDOFF.md) for expected runtimes, cluster use, and
the handoff checklist.

## Licensing notice

This repository contains modifications built on PAFlow. The upstream PAFlow
repository did not include a license when this handoff package was prepared.
The Hugging Face artifacts are publicly available under the owner's release
decision, but permission and an appropriate project license should still be
confirmed before redistributing PAFlow-derived code or checkpoints elsewhere.
