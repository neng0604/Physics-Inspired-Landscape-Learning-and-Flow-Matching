---
pretty_name: LandFlow trajectory supervision
task_categories:
  - other
tags:
  - molecular-generation
  - structure-based-drug-design
  - trajectories
---

# LandFlow trajectory supervision

This repository contains the 3,000-trajectory bank used to train the LandFlow
route potential, the fixed CrossDocked split, atom-number metadata, reference
metric cache, and the PAFlow-ready CrossDocked pocket data required by the
reproduction workflow.

The bank contains 50 recorded states per trajectory (150,000 supervised state
examples). It was built from 100 training pockets using three frozen PAFlow
sampling settings and does not contain the paper test pockets as training
trajectories.

The CrossDocked portion contains the processed LMDB used directly by PAFlow,
the 93-pocket test receptor set, and the compressed pocket10 source files used
to regenerate processed data. The source archive expands to about 15 GB and is
kept compressed because it contains roughly 389,000 small files.

CrossDocked2020 is redistributed under the CC0 1.0 Universal Public Domain
Dedication. See `data/CrossDocked2020_LICENSE.txt` and the original distribution
at https://bits.csb.pitt.edu/files/crossdock2020/.

Use the matching LandFlow GitHub release to download and extract the data:

```bash
python scripts/download_artifacts.py --include-crossdocked --extract-test-set
python scripts/verify_install.py --require-crossdocked
```
