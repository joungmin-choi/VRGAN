"""
VRGAN: a deep generative framework for unsupervised anomaly detection in
multivariate time-series data.

PyTorch re-implementation of the VRNN + GAN anomaly detection model
described in `Time_series_anomaly_detection_v3.pdf` (Choi et al.).
This is a from-scratch conversion of the legacy TensorFlow 1.x prototype
`vae-gan-v6.py`; see README.md for a summary of the behavioral differences
introduced during conversion.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

if device == "cpu":
    # Thread oversubscription on many-core CPU-only machines makes small
    # per-op tensors (this model's sequences are short) dramatically slower
    # than a modest thread count; cap it unless the user has set it already.
    torch.set_num_threads(min(4, os.cpu_count() or 1))


# --------------------------------------------------------------------------
# Anomaly-candidate masking (paper Eq. 1, "Defining the mask based on the
# anomaly candidates"). Two interchangeable strategies are supported so the
# model never needs true anomaly labels:
#   * "sd"      -> per-timepoint Z-score rule
#   * "outlier" -> intersection of DBSCAN and IQR outliers
# mask[i] = 1  -> normal / observed point (kept for the VRNN encoder)
# mask[i] = 0  -> anomaly candidate (hidden from the encoder during training)
# --------------------------------------------------------------------------
def zscore_mask(x, feature_thresh=2.0, frac_thresh=0.01):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    z = (x - mu) / sd
    frac_high_variant = (np.abs(z) > feature_thresh).mean(axis=1)
    is_candidate = frac_high_variant >= frac_thresh

    # Smoothness weight lambda_i (Eq. 9): scaled to [0, 1], with anomalous
    # timepoints (high |Z-score|) receiving a *low* weight so the model is
    # not forced to smooth its output through a candidate anomaly.
    mean_abs_z = np.abs(z).mean(axis=1)
    scaled = mean_abs_z / (mean_abs_z.max() + 1e-8)
    lamda = 1.0 - scaled

    mask = (~is_candidate).astype(np.float32)
    return mask, is_candidate, lamda.astype(np.float32)


def outlier_mask(x, iqr_k=1.5, dbscan_eps=0.5, dbscan_min_samples=5):
    from sklearn.cluster import DBSCAN

    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    iqr = q3 - q1
    lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
    iqr_outlier = ((x < lower) | (x > upper)).any(axis=1)

    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(x)
    dbscan_outlier = labels == -1

    is_candidate = iqr_outlier & dbscan_outlier
    mask = (~is_candidate).astype(np.float32)
    # No Z-score is computed for this strategy; the paper uses a fixed 0.5.
    lamda = np.full(len(x), 0.5, dtype=np.float32)
    return mask, is_candidate, lamda


def compute_candidate_mask(x, method):
    if method == "sd":
        return zscore_mask(x)
    if method == "outlier":
        return outlier_mask(x)
    if method == "none":
        mask = np.ones(len(x), dtype=np.float32)
        return mask, np.zeros(len(x), dtype=bool), np.full(len(x), 0.5, dtype=np.float32)
    raise ValueError(f"Unknown mask method: {method}")


# --------------------------------------------------------------------------
# Windowing + time-decay features
# --------------------------------------------------------------------------
def make_windows(arr, window_size):
    n_windows = len(arr) // window_size
    trimmed = arr[: n_windows * window_size]
    return trimmed.reshape(n_windows, window_size, -1)


def compute_time_decay(mask_windows):
    """Forward/backward gap-since-last-observed-point, per Eq. 12 (converted
    from the dict-of-lists loop in vae-gan-v6.py to plain numpy arrays)."""
    n, t = mask_windows.shape
    forward = np.zeros_like(mask_windows, dtype=np.float32)
    backward = np.zeros_like(mask_windows, dtype=np.float32)
    for i in range(n):
        gap = 0
        for j in range(t):
            if j == 0:
                forward[i, j] = 0
            else:
                gap = 1 if mask_windows[i, j - 1] == 1 else gap + 1
                forward[i, j] = gap
        gap = 0
        for j in range(t - 1, -1, -1):
            if j == t - 1:
                backward[i, j] = 0
            else:
                gap = 1 if mask_windows[i, j + 1] == 1 else gap + 1
                backward[i, j] = gap
    return forward, backward


# --------------------------------------------------------------------------
# Model definition
# --------------------------------------------------------------------------
class FCBN(nn.Module):
    """Fully-connected layer + batch norm applied along the feature axis of
    a (batch, time, feature) tensor. Equivalent of the TF `fc_bn` helper."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        b, t, _ = x.shape
        h = self.fc(x)
        h = self.bn(h.reshape(b * t, -1)).reshape(b, t, -1)
        return h


class Encoder(nn.Module):
    """phi_enc: two FC+activation layers followed by a GRU producing the
    approximate-posterior Gaussian parameters (mu, sigma)."""

    def __init__(self, num_feature, h1=32, h2=16, latent_dim=8, dropout=0.5):
        super().__init__()
        self.fc1 = FCBN(num_feature, h1)
        self.fc2 = FCBN(h1, h2)
        self.gru = nn.GRU(h2, latent_dim * 2, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.latent_dim = latent_dim

    def forward(self, x, mask):
        x_masked = x * mask
        h1 = self.drop(F.elu(self.fc1(x_masked)))
        # NOTE: vae-gan-v6.py computed `en2` but fed `en1` into the GRU,
        # leaving the second FC layer dead code. Both FC layers are wired
        # in sequence here, matching the paper's "two FC layers" encoder.
        h2 = self.drop(torch.tanh(self.fc2(h1)))
        gru_out, _ = self.gru(h2)
        mu = gru_out[..., : self.latent_dim]
        sigma = 1e-6 + F.softplus(gru_out[..., self.latent_dim :])
        return mu, sigma


class Decoder(nn.Module):
    """phi_dec: mirrors the encoder (GRU -> FC -> FC), ending in a sigmoid
    so the reconstruction matches the min-max normalized [0, 1] input."""

    def __init__(self, latent_dim=8, h1=16, h2=32, num_feature=1, dropout=0.5):
        super().__init__()
        self.gru = nn.GRU(latent_dim, h1, batch_first=True)
        self.fc2 = FCBN(h1, h2)
        self.fc_out = FCBN(h2, num_feature)
        self.drop = nn.Dropout(dropout)

    def forward(self, z):
        gru_out, _ = self.gru(z)
        d2 = self.drop(gru_out)
        d3 = self.drop(torch.tanh(self.fc2(d2)))
        de_out = F.elu(self.fc_out(d3))
        decode = torch.sigmoid(de_out)
        return decode, de_out


class Generator(nn.Module):
    """GAN generator: a forward/backward RNN pair over the VRNN latent code,
    combined with a learned exponential time-decay weight per direction
    (Eq. 11-12), following the BRITS-style imputation used in vae-gan-v6.py."""

    def __init__(self, latent_dim=8, hidden=16, num_feature=1):
        super().__init__()
        self.rnn_f = nn.RNN(latent_dim, hidden, batch_first=True)
        self.rnn_b = nn.RNN(latent_dim, hidden, batch_first=True)
        self.out_f = nn.Linear(hidden, num_feature)
        self.out_b = nn.Linear(hidden, num_feature)
        self.timedecay_f = FCBN(1, 1)
        self.timedecay_b = FCBN(1, 1)

    def forward(self, x, z, mask, td_f, td_b):
        f_out, _ = self.rnn_f(z)
        z_rev = torch.flip(z, dims=[1])
        b_out, _ = self.rnn_b(z_rev)
        b_out = torch.flip(b_out, dims=[1])

        output_f = torch.sigmoid(self.out_f(f_out))
        output_b = torch.sigmoid(self.out_b(b_out))
        difference_fb = torch.abs(output_f - output_b)

        lamda_f = torch.exp(-torch.clamp(F.leaky_relu(self.timedecay_f(td_f)), min=0))
        lamda_b = torch.exp(-torch.clamp(F.leaky_relu(self.timedecay_b(td_b)), min=0))

        x_estimated = lamda_f * output_f + lamda_b * output_b
        x_real_mask = x * mask
        x_estimated_mask = x_estimated * (1 - mask)
        x_imputed = x_real_mask + x_estimated_mask
        return x_estimated, x_imputed, x_real_mask, x_estimated_mask, difference_fb


class Discriminator(nn.Module):
    """One-layer RNN followed by two FC layers ending in a sigmoid."""

    def __init__(self, num_feature, hidden=16):
        super().__init__()
        self.rnn = nn.LSTM(num_feature, hidden, batch_first=True)
        self.fc1 = nn.Linear(hidden, 2)
        self.fc2 = FCBN(2, 1)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        h = self.fc1(rnn_out)
        logits = self.fc2(h)
        pred = torch.sigmoid(logits)
        return pred, logits


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------
def vrnn_recon_kl(x, de_out, mu, sigma):
    """Timestep-wise variational lower bound (Eq. 8), vectorized.

    vae-gan-v6.py computed this with a double loop that summed, for every
    prefix length i, the loss over [0, i) and accumulated across i. That is
    algebraically identical to weighting timestep t's loss by (T - 1 - t),
    which is what is done here directly.

    NOTE (bugfix): the legacy script set
        vae_cost = -1 * marginal_likelihood + KL_divergence
    where `marginal_likelihood` was itself already the (positive)
    reconstruction *loss* from `sigmoid_cross_entropy_with_logits` -- i.e.
    the negative log-likelihood. The extra unary minus therefore made the
    optimizer *maximize* reconstruction error. Both terms are minimized here.
    """
    b, t, _ = x.shape
    bce = F.binary_cross_entropy_with_logits(de_out, x, reduction="none").sum(-1)
    kl = 0.5 * (mu.pow(2) + sigma.pow(2) - torch.log(1e-8 + sigma.pow(2)) - 1).sum(-1)
    weights = torch.arange(t - 1, -1, -1, device=x.device, dtype=x.dtype)
    recon_loss = (bce * weights).sum(-1).mean()
    kl_loss = (kl * weights).sum(-1).mean()
    return recon_loss, kl_loss


def bernoulli_kl(p, q, eps=1e-6):
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).sum(-1)


def smoothness_loss(decode, lamda):
    """Eq. 9. `decode` holds the Bernoulli reconstruction probabilities, so
    the KL divergence between consecutive timepoints is computed directly
    from those probabilities rather than feeding raw logits into a
    Keras `KLDivergence` op (as vae-gan-v6.py did, which is only valid for
    non-negative probability-like inputs)."""
    kl_terms = bernoulli_kl(decode[:, :-1, :], decode[:, 1:, :])
    return (lamda[:, :-1] * kl_terms).mean()


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_vrnn(encoder, decoder, x, mask, lamda, epochs, lr, smooth_reg, log_every=100):
    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        opt.zero_grad()
        mu, sigma = encoder(x, mask)
        z = mu + sigma * torch.randn_like(mu)
        decode, de_out = decoder(z)
        recon_loss, kl_loss = vrnn_recon_kl(x, de_out, mu, sigma)
        loss = recon_loss + kl_loss
        if smooth_reg:
            loss = loss + smoothness_loss(decode, lamda)
        loss.backward()
        opt.step()
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[VRNN] epoch {epoch:5d}  recon={recon_loss.item():.4f}  kl={kl_loss.item():.4f}  loss={loss.item():.4f}")


def train_gan(generator, discriminator, x, z, mask, td_f, td_b, decode,
              epochs, disc_steps, gen_steps, lr_g, lr_d, pretrain_steps=10, log_every=100):
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr_g)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()
    decode = decode.detach()

    def gen_losses():
        generated_x, imputed_x, x_real_mask, x_estimated_mask, diff_fb = generator(x, z, mask, td_f, td_b)
        _, fake_logits = discriminator(x_estimated_mask)
        gen_loss = bce(fake_logits, torch.ones_like(fake_logits))
        recon_loss = (x_real_mask - generated_x * mask).abs().mean()
        consistency_loss = diff_fb.mean()
        g_loss = gen_loss + recon_loss + consistency_loss
        return g_loss, generated_x, imputed_x, x_real_mask, x_estimated_mask

    generator.train()
    for _ in range(pretrain_steps):
        opt_g.zero_grad()
        g_loss, *_ = gen_losses()
        g_loss.backward()
        opt_g.step()

    for epoch in range(epochs):
        for _ in range(disc_steps):
            generator.eval()
            with torch.no_grad():
                _, _, x_real_mask, x_estimated_mask, _ = generator(x, z, mask, td_f, td_b)
            discriminator.train()
            opt_d.zero_grad()
            _, real_logits = discriminator(x_real_mask)
            _, fake_logits = discriminator(x_estimated_mask)
            _, v_fake_logits = discriminator(decode)
            d_loss = (
                bce(real_logits, torch.ones_like(real_logits))
                + bce(fake_logits, torch.zeros_like(fake_logits))
                + bce(v_fake_logits, torch.zeros_like(v_fake_logits))
            )
            d_loss.backward()
            opt_d.step()

        generator.train()
        for _ in range(gen_steps):
            opt_g.zero_grad()
            g_loss, generated_x, imputed_x, x_real_mask, x_estimated_mask = gen_losses()
            g_loss.backward()
            opt_g.step()

        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"[GAN]  epoch {epoch:5d}  g_loss={g_loss.item():.4f}  d_loss={d_loss.item():.4f}")


# --------------------------------------------------------------------------
# Reconstruction / anomaly scoring
# --------------------------------------------------------------------------
@torch.no_grad()
def reconstruct(encoder, decoder, generator, x, mask, td_f, td_b):
    encoder.eval()
    decoder.eval()
    generator.eval()
    mu, _ = encoder(x, mask)
    decode, _ = decoder(mu)
    generated_x, _, _, _, _ = generator(x, mu, mask, td_f, td_b)
    return decode, generated_x


def anomaly_score(x, vrnn_recon, gan_recon):
    """Eq. 17: s_i = ||x_i - x_hat_i|| + ||x_i - x_tilde_i||."""
    s_vrnn = torch.sqrt(((x - vrnn_recon) ** 2).sum(-1) + 1e-12)
    s_gan = torch.sqrt(((x - gan_recon) ** 2).sum(-1) + 1e-12)
    return s_vrnn + s_gan


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="VRGAN: unsupervised time-series anomaly detection")
    p.add_argument("data_dir", help="Directory containing train.csv and test.csv")
    p.add_argument("result_dir", help="Directory to write results into")
    p.add_argument("window_size", type=int, help="Number of timepoints per training window")
    p.add_argument("mask_method", choices=["sd", "outlier", "none"],
                   help="Anomaly-candidate identification strategy for masking")
    p.add_argument("vrnn_epochs", type=int, help="Number of VRNN (Phase 1) training epochs")
    p.add_argument("gan_epochs", type=int, help="Number of GAN (Phase 2) training epochs")
    p.add_argument("--train_ratio", type=float, default=0.6,
                   help="Fraction of train.csv windows used for VRNN/GAN training; the rest is held out for threshold selection")
    p.add_argument("--disc_steps", type=int, default=5)
    p.add_argument("--gen_steps", type=int, default=1)
    p.add_argument("--pretrain_steps", type=int, default=10)
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--lr_vrnn", type=float, default=1e-4)
    p.add_argument("--lr_g", type=float, default=1e-3)
    p.add_argument("--lr_d", type=float, default=1e-3)
    p.add_argument("--no_smooth_reg", action="store_true", help="Disable the smoothness regularizer (ablation)")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.result_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"), index_col=0)
    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"), index_col=0)

    label_col = "label"
    test_labels = None
    if label_col in test_df.columns:
        test_labels = test_df[label_col].values.astype(int)
        test_df = test_df.drop(columns=[label_col])

    feature_cols = train_df.columns.tolist()
    num_feature = len(feature_cols)

    # Min-max normalization fit on the training set only.
    train_min = train_df.min(axis=0).values
    train_max = train_df.max(axis=0).values
    scale = np.where((train_max - train_min) > 1e-8, train_max - train_min, 1.0)

    train_raw = ((train_df.values - train_min) / scale).astype(np.float32)
    test_raw = ((test_df.values - train_min) / scale).astype(np.float32)
    test_raw = np.clip(test_raw, 0.0, 1.0)

    # Anomaly-candidate masking is computed on the full continuous series
    # (more meaningful than per-window statistics) and only then reshaped
    # into windows to match the model's fixed-length input.
    train_mask_flat, train_candidate_flat, train_lamda_flat = compute_candidate_mask(train_raw, args.mask_method)
    test_mask_flat, test_candidate_flat, test_lamda_flat = compute_candidate_mask(test_raw, args.mask_method)
    # Eq. 9: lambda is fixed at 0.5 for the held-out/test data.
    test_lamda_flat[:] = 0.5

    x_windows = make_windows(train_raw, args.window_size)
    mask_windows = make_windows(train_mask_flat[:, None], args.window_size)[..., 0]
    lamda_windows = make_windows(train_lamda_flat[:, None], args.window_size)[..., 0]

    x_test_windows = make_windows(test_raw, args.window_size)
    mask_test_windows = make_windows(test_mask_flat[:, None], args.window_size)[..., 0]
    lamda_test_windows = make_windows(test_lamda_flat[:, None], args.window_size)[..., 0]

    n_windows = len(x_windows)
    n_train = max(1, int(round(n_windows * args.train_ratio)))
    n_train = min(n_train, n_windows - 1) if n_windows > 1 else n_windows

    perm = np.arange(n_windows)  # windows kept in temporal order (no shuffling of time)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    if len(val_idx) == 0:
        val_idx = train_idx

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)

    x_train = to_tensor(x_windows[train_idx])
    mask_train = to_tensor(mask_windows[train_idx])[..., None]
    lamda_train = to_tensor(lamda_windows[train_idx])

    x_val = to_tensor(x_windows[val_idx])
    mask_val = to_tensor(mask_windows[val_idx])[..., None]

    td_f_train, td_b_train = compute_time_decay(mask_windows[train_idx])
    td_f_train = to_tensor(td_f_train)[..., None]
    td_b_train = to_tensor(td_b_train)[..., None]

    td_f_val, td_b_val = compute_time_decay(mask_windows[val_idx])
    td_f_val = to_tensor(td_f_val)[..., None]
    td_b_val = to_tensor(td_b_val)[..., None]

    x_test = to_tensor(x_test_windows)
    mask_test = to_tensor(mask_test_windows)[..., None]
    td_f_test, td_b_test = compute_time_decay(mask_test_windows)
    td_f_test = to_tensor(td_f_test)[..., None]
    td_b_test = to_tensor(td_b_test)[..., None]

    # ---- Build the model ----
    encoder = Encoder(num_feature, latent_dim=args.latent_dim).to(device)
    decoder = Decoder(latent_dim=args.latent_dim, num_feature=num_feature).to(device)
    generator = Generator(latent_dim=args.latent_dim, num_feature=num_feature).to(device)
    discriminator = Discriminator(num_feature).to(device)

    # ---- Phase 1: train VRNN ----
    train_vrnn(encoder, decoder, x_train, mask_train, lamda_train,
               epochs=args.vrnn_epochs, lr=args.lr_vrnn, smooth_reg=not args.no_smooth_reg)

    # ---- Freeze VRNN, sample the latent code once, train the GAN module ----
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        mu_train, sigma_train = encoder(x_train, mask_train)
        z_train = mu_train + sigma_train * torch.randn_like(mu_train)
        decode_train, _ = decoder(z_train)
    for p in encoder.parameters():
        p.requires_grad_(False)
    for p in decoder.parameters():
        p.requires_grad_(False)

    train_gan(generator, discriminator, x_train, z_train.detach(), mask_train,
              td_f_train, td_b_train, decode_train,
              epochs=args.gan_epochs, disc_steps=args.disc_steps, gen_steps=args.gen_steps,
              lr_g=args.lr_g, lr_d=args.lr_d, pretrain_steps=args.pretrain_steps)

    # ---- Threshold selection on the held-out validation windows ----
    vrnn_recon_val, gan_recon_val = reconstruct(encoder, decoder, generator, x_val, mask_val, td_f_val, td_b_val)
    val_scores = anomaly_score(x_val, vrnn_recon_val, gan_recon_val)
    normal_scores = val_scores[mask_val[..., 0] == 1]
    threshold = normal_scores.max().item() if normal_scores.numel() > 0 else val_scores.max().item()
    print(f"Anomaly threshold (max reconstruction error among normal validation candidates): {threshold:.6f}")

    # ---- Score the test set ----
    vrnn_recon_test, gan_recon_test = reconstruct(encoder, decoder, generator, x_test, mask_test, td_f_test, td_b_test)
    test_scores = anomaly_score(x_test, vrnn_recon_test, gan_recon_test)

    n_used = x_test_windows.shape[0] * args.window_size
    test_index = test_df.index.tolist()[:n_used]

    scores_flat = test_scores.reshape(-1).cpu().numpy()
    pred_flat = (scores_flat > threshold).astype(int)
    candidate_flat = test_candidate_flat[:n_used]

    def denorm(arr_windows):
        flat = arr_windows.reshape(-1, num_feature).cpu().numpy()
        return flat * scale + train_min

    vrnn_recon_df = pd.DataFrame(denorm(vrnn_recon_test), columns=feature_cols, index=test_index)
    vrnn_recon_df.to_csv(os.path.join(args.result_dir, "vrnn_reconstruction_test.csv"))

    gan_recon_df = pd.DataFrame(denorm(gan_recon_test), columns=feature_cols, index=test_index)
    gan_recon_df.to_csv(os.path.join(args.result_dir, "gan_reconstruction_test.csv"))

    score_df = pd.DataFrame(
        {
            "anomaly_score": scores_flat,
            "is_anomaly_candidate": candidate_flat,
            "predicted_anomaly": pred_flat,
        },
        index=test_index,
    )
    if test_labels is not None:
        score_df["label"] = test_labels[:n_used]
    score_df.to_csv(os.path.join(args.result_dir, "anomaly_score_test.csv"))

    with open(os.path.join(args.result_dir, "threshold.txt"), "w") as f:
        f.write(f"{threshold:.6f}\n")

    if test_labels is not None:
        from sklearn.metrics import f1_score, roc_auc_score
        y_true = test_labels[:n_used]
        try:
            auc = roc_auc_score(y_true, scores_flat)
        except ValueError:
            auc = float("nan")
        f1 = f1_score(y_true, pred_flat, average="weighted")
        with open(os.path.join(args.result_dir, "metrics.txt"), "w") as f:
            f.write(f"AUC: {auc:.4f}\nWeighted F1: {f1:.4f}\n")
        print(f"AUC: {auc:.4f}  Weighted F1: {f1:.4f}")

    print(f"Done. Results written to {args.result_dir}")


if __name__ == "__main__":
    main()
