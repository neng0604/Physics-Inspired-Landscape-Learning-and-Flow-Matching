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
route potential, plus the fixed CrossDocked split, atom-number metadata, and
reference metric cache required by the reproduction workflow.

The bank contains 50 recorded states per trajectory (150,000 supervised state
examples). It was built from 100 training pockets using three frozen PAFlow
sampling settings and does not contain the paper test pockets as training
trajectories.

The processed CrossDocked LMDB and receptor files are not included. Obtain them
through the original PAFlow/CrossDocked distribution and use the matching
GitHub release for loading instructions.
