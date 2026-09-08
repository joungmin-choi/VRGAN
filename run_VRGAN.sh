#!/bin/bash

dataDir="./example"	# Dataset directory containing train.csv and test.csv
resDir="./results"	# Result directory to save results
window_size=20		# Number of timepoints per training window
mask_method="sd"	# Anomaly-candidate masking strategy: sd | outlier | none
vrnn_epochs=2000	# Phase 1 (VRNN) training epochs
gan_epochs=1000		# Phase 2 (GAN) training epochs

python3 VRGAN.py $dataDir $resDir $window_size $mask_method $vrnn_epochs $gan_epochs
