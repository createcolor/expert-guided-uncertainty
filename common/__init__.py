"""Task-agnostic building blocks shared across classification, segmentation and QA.

The submodules here never import task code and hold no task-specific assumptions:

  - ``backends``    one interface over numpy arrays and torch tensors
  - ``estimators``  uncertainty estimators over ``[members, units, classes]`` arrays
  - ``rejection``   the rejection-curve driver, parameterised by a scoring callback

A "unit" is whatever the leading non-member axis indexes -- a sample for
classification/QA, a pixel for segmentation. The math does not care which.
"""
