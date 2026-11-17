#!/usr/bin/env bash
# Option A (used to build this project): public GitHub mirror of the CWRU data
#   (commit 0d07cb1caf0a5a68a955c2f43cfc21b9a208406a, ~890 MB)
# Option B: official site https://engineering.case.edu/bearingdatacenter/download-data-file
#   download the 12k Drive End, 12k Fan End and Normal Baseline .mat files into data/CWRU-dataset/
set -e
mkdir -p data
git clone --depth 1 https://github.com/s-whynot/CWRU-dataset.git data/CWRU-dataset
