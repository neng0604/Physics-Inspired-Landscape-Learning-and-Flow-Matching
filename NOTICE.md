# Source and redistribution notice

LandFlow is implemented on top of the PAFlow research code:

- PAFlow repository: https://github.com/CMACH508/PAFlow
- PAFlow paper: "Prior-Guided Flow Matching for Target-Aware Molecule Design
  with Learnable Atom Number," NeurIPS 2025.

The frozen PAFlow model and atom-number predictor are redistributed only in the
private handoff repositories described here. At the time this package was
prepared, the upstream PAFlow repository did not contain a LICENSE file.
Public redistribution therefore requires confirmation from the upstream
authors or another documented legal basis.

CrossDocked2020 data is not committed to GitHub. The public Hugging Face dataset
repository contains the PAFlow-ready pocket10 source archive, processed
LMDB, and test receptor files under the dataset's CC0 1.0 terms. The official
full CrossDocked2020 release remains available from its original distribution.
