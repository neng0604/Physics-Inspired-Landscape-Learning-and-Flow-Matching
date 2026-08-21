---
library_name: pytorch
pipeline_tag: other
tags:
  - flow-matching
  - structure-based-drug-design
  - molecular-generation
  - pytorch
---

# LandFlow checkpoints

This repository stores the frozen PAFlow base checkpoint, its atom-number
predictor, and the LandFlow route-potential checkpoint used in the main
CrossDocked2020 experiment.

The LandFlow potential is a pairwise scalar model with 115,842 parameters. It
predicts a trajectory-derived future-quality cost and supplies a normalized
negative coordinate gradient during sampling. The PAFlow parameters remain
frozen.

Use these files with the matching GitHub release. SHA-256 checksums are listed
in `artifacts.json` in that repository.

LandFlow is implemented on top of the PAFlow research code and includes frozen
PAFlow checkpoints. The upstream repository did not contain a license when this
release was prepared; verify the upstream redistribution terms before mirroring
these files.

This model is for research use in molecular generation. Generated structures
must not be interpreted as experimentally validated binders or medical advice.
