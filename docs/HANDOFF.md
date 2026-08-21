# Handoff Guide

## What a new maintainer should receive

- access to the private GitHub repository;
- the public Hugging Face model and dataset repositories;
- access to a Linux GPU node with CUDA 12.1-compatible drivers;
- at least 10 GB of local disk for the standard handoff, or about 25 GB if the
  compressed pocket10 source archive is also extracted;
- a Slurm account/partition only if the local cluster uses Slurm.

No GitHub or Hugging Face token should be stored in this repository. Hugging
Face authentication is optional for these public artifacts.

## Recommended first-day check

```bash
conda env create -f environment.yml
conda activate landflow
export HF_MODEL_REPO=NYCU-MLLab/LandFlow
export HF_DATA_REPO=NYCU-MLLab/LandFlow-data
python scripts/download_artifacts.py --include-crossdocked --extract-test-set
python scripts/verify_install.py --require-crossdocked
EPOCHS=1 MAX_TRAIN_BATCHES=2 VAL_MAX_BATCHES=2 \
  OUTPUT_DIR=outputs/training_smoke bash workflows/03_train_potential.sh
DATA_START=0 DATA_END=0 NUM_SAMPLES=2 bash workflows/04_sample_landflow.sh
DOCKING_MODE=none bash workflows/05_evaluate.sh
```

The first sampling check requires the processed CrossDocked LMDB. Vina docking
also requires `data/test_set/` and the Vina/OpenBabel tools from the environment.

## Main outputs

```text
outputs/trajectories/                 generated train trajectories
artifacts/trajectory_bank.pt         potential-training bank
outputs/landflow_training/best.pt    newly trained potential
results/landflow/test/result_*.pt    generated test ligands and traces
results/landflow/test/eval_results/  reconstruction and docking metrics
```

## Expected scale on the original cluster

The one-pocket/two-ligand smoke run should finish far sooner than the full
experiment and is the required first check. Full timings depend on GPU model,
filesystem load, and Vina CPU parallelism. Preserve the original resource
envelope when planning the first full rerun:

- trajectory collection: one GPU, up to 36 hours per gamma job;
- bank construction: one GPU, up to 12 hours;
- potential training: one GPU, up to 12 hours;
- full100 sampling and evaluation: one GPU plus CPU docking workers, up to 4
  hours in the original job template.

These are scheduler limits, not guaranteed wall-clock runtimes. Record actual
runtimes in the repository after the first clean rerun.

## Ownership checklist

- protect the default branch;
- add the junior maintainer with Write access;
- confirm PAFlow code and checkpoint redistribution terms before changing the
  release scope; CrossDocked2020 itself is CC0 1.0;
- enable Git LFS/Xet through Hugging Face for checkpoint and bank files;
- create a tagged release only after the one-pocket and full100 checks pass;
- record the Git commit and artifact SHA-256 values used for every paper result.
