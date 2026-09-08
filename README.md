# VRGAN: a deep generative framework for unsupervised anomaly detection in multivariate time-series data

PyTorch implementation of VRGAN, a fully-unsupervised time-series anomaly
detection model that combines a variational recurrent neural network (VRNN)
with a GAN-based imputation module. Unlike most reconstruction-based anomaly
detectors, VRGAN does not assume the training set only contains normal
timepoints: candidate anomalies are identified and masked automatically
(via a Z-score rule or DBSCAN/IQR outlier detection) before training, so no
true anomaly labels are required.

See `Time_series_anomaly_detection_v3.pdf` for the full method description.
This repository is a from-scratch PyTorch conversion of an earlier
TensorFlow 1.x prototype (`vae-gan-v6.py`); see "Notes on the PyTorch
conversion" below for what changed.

## Requirements
* Python (>= 3.7)
* PyTorch (>= v1.10.0)
* Other python packages: numpy (>=1.19.1), pandas (>=1.1.1), scikit-learn (>=0.24.0)

## Installation
Clone the repository, then run either:
```
pip install -r requirements.txt
```
or:
```
pip install .
```

## Inputs
[Note!] An example dataset is provided in `./example/`.

#### Time-series data (train / test)
* Row: timepoint, Column: feature.
* The first column is the timepoint index (kept as the row index when read with `index_col=0`).
* `train.csv` is used for both VRNN/GAN training and (a held-out slice of it) threshold selection; no anomaly labels are needed.
* `test.csv` is the series to be scored for anomalies. It may optionally include a `label` column (0 = normal, 1 = anomaly) — if present, AUC and weighted F1 are reported in `metrics.txt`.
* File names must be `train.csv` and `test.csv`.
* Example: `./example/train.csv`, `./example/test.csv`

## How to run (example)
1. Clone the repository, move into it, and edit `run_VRGAN.sh` to point at your dataset and hyperparameters.
2. Run:
```
chmod +x run_VRGAN.sh
./run_VRGAN.sh
```
Running it as-is (no edits) trains on the bundled synthetic example dataset.

3. Results are written to the `results` directory:
   * `vrnn_reconstruction_test.csv` — VRNN reconstruction of the test set
   * `gan_reconstruction_test.csv` — GAN-generated reconstruction of the test set
   * `anomaly_score_test.csv` — per-timepoint anomaly score, candidate-mask flag, and predicted label
   * `threshold.txt` — the anomaly score threshold selected from the held-out validation candidates
   * `metrics.txt` — AUC / weighted F1, only written if `test.csv` has a `label` column

## Usage
```
python3 VRGAN.py <data_dir> <result_dir> <window_size> <mask_method> <vrnn_epochs> <gan_epochs> [options]
```
* `mask_method`: `sd` (Z-score based candidates), `outlier` (DBSCAN ∩ IQR based candidates), or `none` (no masking, all points treated as observed)

Optional flags: `--train_ratio` (default 0.6), `--disc_steps` (5), `--gen_steps` (1), `--pretrain_steps` (10), `--latent_dim` (8), `--lr_vrnn` (1e-4), `--lr_g` / `--lr_d` (1e-3), `--no_smooth_reg`, `--seed` (42).

## Model summary
1. **Identify anomaly candidates** — mask timepoints flagged by the Z-score or outlier rule so they don't bias training.
2. **Phase 1 (VRNN)** — an encoder/decoder pair (2 FC layers + GRU) is trained as a variational recurrent autoencoder, with an optional smoothness regularizer that is down-weighted at candidate anomalies.
3. **Phase 2 (GAN)** — with the VRNN frozen, a bidirectional-RNN generator with a learned time-decay combination is trained adversarially (discriminator:generator step ratio 5:1) against a one-layer RNN discriminator.
4. **Score & threshold** — the anomaly score is `||x - VRNN_reconstruction|| + ||x - GAN_reconstruction||`; the threshold is the highest score observed among non-candidate (normal) points in the held-out validation split.


## Contact
If you have any questions or problems, please contact **joungmin AT vt.edu**.
