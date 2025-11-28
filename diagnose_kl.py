import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pickle
from train import load_data, create_model

# Load data
print("Loading data...")
adjacency_tensor, feature_tensor, qed_tensor = load_data("data")
batch_size_total, bond_dim, num_atoms, _ = adjacency_tensor.shape
_, _, atom_dim = feature_tensor.shape
adjacency_shape = (bond_dim, num_atoms, num_atoms)
feature_shape = (num_atoms, atom_dim)

# Create model
print("Creating model...")
vae = create_model(435, adjacency_shape, feature_shape)

# Load checkpoint
checkpoint_path = "checkpoints/checkpoint_best.npz"
print(f"Loading checkpoint: {checkpoint_path}")
try:
    vae.load_weights(checkpoint_path)
    print("  ✓ Checkpoint loaded")
except Exception as e:
    print(f"  Error loading checkpoint: {e}")
    exit(1)

# Sample a batch
batch_size = 20
indices = np.random.choice(batch_size_total, batch_size, replace=False)
adj_batch = adjacency_tensor[indices.tolist()]
feat_batch = feature_tensor[indices.tolist()]

# Forward pass
mu, logvar, adj_gen, feat_gen, qed_pred = vae((adj_batch, feat_batch))

# Convert to numpy
mu_np = np.array(mu)
logvar_np = np.array(logvar)

print("\n" + "="*70)
print("ENCODER OUTPUT ANALYSIS")
print("="*70)

print(f"\nMu (latent mean) statistics:")
print(f"  Shape: {mu_np.shape}")
print(f"  Min: {mu_np.min():.6f}")
print(f"  Max: {mu_np.max():.6f}")
print(f"  Mean: {mu_np.mean():.6f}")
print(f"  Std: {mu_np.std():.6f}")
print(f"  Mean absolute value: {np.abs(mu_np).mean():.6f}")

print(f"\nLogvar (latent log variance) statistics:")
print(f"  Shape: {logvar_np.shape}")
print(f"  Min: {logvar_np.min():.6f}")
print(f"  Max: {logvar_np.max():.6f}")
print(f"  Mean: {logvar_np.mean():.6f}")
print(f"  Std: {logvar_np.std():.6f}")

# Compute variance
var_np = np.exp(logvar_np)
print(f"\nVariance (exp(logvar)) statistics:")
print(f"  Min: {var_np.min():.6f}")
print(f"  Max: {var_np.max():.6f}")
print(f"  Mean: {var_np.mean():.6f}")
print(f"  Std: {var_np.std():.6f}")

# Compute KL loss manually
kl_per_sample = -0.5 * np.sum(1 + logvar_np - mu_np**2 - np.exp(logvar_np), axis=1)
kl_mean = kl_per_sample.mean()
print(f"\nKL Loss (computed):")
print(f"  Mean: {kl_mean:.6f}")
print(f"  Min: {kl_per_sample.min():.6f}")
print(f"  Max: {kl_per_sample.max():.6f}")
print(f"  Std: {kl_per_sample.std():.6f}")

# Check distribution of logvar values
print(f"\nLogvar distribution:")
logvar_flat = logvar_np.flatten()
print(f"  Values < -5: {(logvar_flat < -5).sum()} ({(logvar_flat < -5).sum() / len(logvar_flat) * 100:.1f}%)")
print(f"  Values < -2: {(logvar_flat < -2).sum()} ({(logvar_flat < -2).sum() / len(logvar_flat) * 100:.1f}%)")
print(f"  Values < 0: {(logvar_flat < 0).sum()} ({(logvar_flat < 0).sum() / len(logvar_flat) * 100:.1f}%)")
print(f"  Values < 1: {(logvar_flat < 1).sum()} ({(logvar_flat < 1).sum() / len(logvar_flat) * 100:.1f}%)")
print(f"  Values < 2: {(logvar_flat < 2).sum()} ({(logvar_flat < 2).sum() / len(logvar_flat) * 100:.1f}%)")

# Check if logvar is collapsing
print(f"\n" + "="*70)
print("POSTERIOR COLLAPSE DIAGNOSIS")
print("="*70)

# If logvar is very negative, variance is near zero
very_low_var = (var_np < 0.01).sum()
print(f"\nDimensions with variance < 0.01: {very_low_var} / {var_np.size} ({very_low_var/var_np.size*100:.1f}%)")

# If mu is near zero, mean is collapsing
mu_near_zero = (np.abs(mu_np) < 0.01).sum()
print(f"Dimensions with |mu| < 0.01: {mu_near_zero} / {mu_np.size} ({mu_near_zero/mu_np.size*100:.1f}%)")

# Check encoder weights
print(f"\n" + "="*70)
print("ENCODER WEIGHT ANALYSIS")
print("="*70)

encoder = vae.encoder
fc_mu = encoder.fc_mu
fc_logvar = encoder.fc_logvar

mu_weight = np.array(fc_mu.weight)
mu_bias = np.array(fc_mu.bias)
logvar_weight = np.array(fc_logvar.weight)
logvar_bias = np.array(fc_logvar.bias)

print(f"\nfc_mu weights:")
print(f"  Shape: {mu_weight.shape}")
print(f"  Mean: {mu_weight.mean():.6f}")
print(f"  Std: {mu_weight.std():.6f}")
print(f"  Min: {mu_weight.min():.6f}")
print(f"  Max: {mu_weight.max():.6f}")

print(f"\nfc_mu bias:")
print(f"  Mean: {mu_bias.mean():.6f}")
print(f"  Std: {mu_bias.std():.6f}")

print(f"\nfc_logvar weights:")
print(f"  Shape: {logvar_weight.shape}")
print(f"  Mean: {logvar_weight.mean():.6f}")
print(f"  Std: {logvar_weight.std():.6f}")
print(f"  Min: {logvar_weight.min():.6f}")
print(f"  Max: {logvar_weight.max():.6f}")

print(f"\nfc_logvar bias:")
print(f"  Mean: {logvar_bias.mean():.6f}")
print(f"  Std: {logvar_bias.std():.6f}")
print(f"  Expected: 2.0 (initialized value)")

# Check if bias has been driven down
if logvar_bias.mean() < 0:
    print(f"\n⚠ WARNING: logvar bias has been driven negative!")
    print(f"  This suggests the KL loss is pushing logvar to collapse")
    print(f"  Initial bias was 2.0, now it's {logvar_bias.mean():.6f}")

print()

