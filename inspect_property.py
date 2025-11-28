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

# Inspect property head weights
print("\n" + "="*70)
print("PROPERTY HEAD INSPECTION")
print("="*70)

property_head = vae.property
print(f"\nProperty head architecture:")
print(f"  Input dim: {property_head.weight.shape[1]}")
print(f"  Output dim: {property_head.weight.shape[0]}")

# Check weights
weight = np.array(property_head.weight)
bias = np.array(property_head.bias)

print(f"\nProperty head weights:")
print(f"  Weight shape: {weight.shape}")
print(f"  Weight stats: min={weight.min():.6f}, max={weight.max():.6f}, mean={weight.mean():.6f}, std={weight.std():.6f}")
print(f"  Bias: {bias[0]:.6f}")

# Sample a batch and inspect mu values
print("\n" + "="*70)
print("LATENT SPACE (mu) INSPECTION")
print("="*70)

batch_size = 20
indices = np.random.choice(batch_size_total, batch_size, replace=False)
adj_batch = adjacency_tensor[indices.tolist()]
feat_batch = feature_tensor[indices.tolist()]
qed_batch = qed_tensor[indices.tolist()]

# Forward pass
mu, logvar, adj_gen, feat_gen, qed_pred = vae((adj_batch, feat_batch))

# Convert to numpy
mu_np = np.array(mu)
logvar_np = np.array(logvar)
qed_pred_np = np.array(qed_pred).flatten()
qed_true_np = np.array(qed_batch).flatten()

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

# Check variance of mu across samples
mu_std_per_dim = mu_np.std(axis=0)
print(f"\nMu standard deviation per dimension:")
print(f"  Mean std across dims: {mu_std_per_dim.mean():.6f}")
print(f"  Min std: {mu_std_per_dim.min():.6f}")
print(f"  Max std: {mu_std_per_dim.max():.6f}")
print(f"  Number of dims with std < 0.01: {(mu_std_per_dim < 0.01).sum()}")
print(f"  Number of dims with std < 0.1: {(mu_std_per_dim < 0.1).sum()}")

# Check property predictions
print("\n" + "="*70)
print("PROPERTY PREDICTION INSPECTION")
print("="*70)

print(f"\nProperty predictions:")
print(f"  Shape: {qed_pred_np.shape}")
print(f"  Min: {qed_pred_np.min():.6f}")
print(f"  Max: {qed_pred_np.max():.6f}")
print(f"  Mean: {qed_pred_np.mean():.6f}")
print(f"  Std: {qed_pred_np.std():.6f}")

print(f"\nTrue QED values:")
print(f"  Min: {qed_true_np.min():.6f}")
print(f"  Max: {qed_true_np.max():.6f}")
print(f"  Mean: {qed_true_np.mean():.6f}")
print(f"  Std: {qed_true_np.std():.6f}")

# Check property logits before sigmoid
property_logits = property_head(mu)
property_logits_np = np.array(property_logits).flatten()

print(f"\nProperty logits (before sigmoid):")
print(f"  Min: {property_logits_np.min():.6f}")
print(f"  Max: {property_logits_np.max():.6f}")
print(f"  Mean: {property_logits_np.mean():.6f}")
print(f"  Std: {property_logits_np.std():.6f}")

# Check correlation
correlation = np.corrcoef(qed_pred_np, qed_true_np)[0, 1]
mae = np.mean(np.abs(qed_pred_np - qed_true_np))
print(f"\nPrediction quality:")
print(f"  MAE: {mae:.6f}")
print(f"  Correlation: {correlation:.6f}")

# Check if mu values are too similar (posterior collapse)
print("\n" + "="*70)
print("POSTERIOR COLLAPSE CHECK")
print("="*70)

# Compute KL loss manually
kl_loss = -0.5 * np.sum(1 + logvar_np - mu_np**2 - np.exp(logvar_np), axis=1)
kl_loss_mean = kl_loss.mean()
print(f"\nKL loss (computed from mu/logvar):")
print(f"  Mean: {kl_loss_mean:.6f}")
print(f"  Min: {kl_loss.min():.6f}")
print(f"  Max: {kl_loss.max():.6f}")

# Check if mu is collapsing to zero
mu_norm = np.linalg.norm(mu_np, axis=1)
print(f"\nMu L2 norm per sample:")
print(f"  Mean: {mu_norm.mean():.6f}")
print(f"  Min: {mu_norm.min():.6f}")
print(f"  Max: {mu_norm.max():.6f}")

# Check variance of mu across samples (should be high if model is using latent space)
mu_variance = mu_np.var(axis=0).mean()
print(f"\nAverage variance of mu across samples: {mu_variance:.6f}")
if mu_variance < 0.01:
    print("  ⚠ WARNING: Very low variance - mu is collapsing!")
elif mu_variance < 0.1:
    print("  ⚠ WARNING: Low variance - mu may be collapsing")

# Check if property head is using the latent space
print("\n" + "="*70)
print("PROPERTY HEAD ANALYSIS")
print("="*70)

# Compute property prediction manually
property_logits_manual = mu_np @ weight.T + bias
property_pred_manual = 1 / (1 + np.exp(-property_logits_manual))

print(f"\nProperty prediction breakdown:")
print(f"  Property logits = mu @ W.T + b")
print(f"  mu contribution range: [{np.min(mu_np @ weight.T):.6f}, {np.max(mu_np @ weight.T):.6f}]")
print(f"  Bias: {bias[0]:.6f}")
print(f"  Final logits range: [{property_logits_manual.min():.6f}, {property_logits_manual.max():.6f}]")

# Check if weight magnitudes are too small
weight_magnitude = np.linalg.norm(weight)
print(f"\nProperty head weight magnitude: {weight_magnitude:.6f}")
if weight_magnitude < 0.1:
    print("  ⚠ WARNING: Property head weights are very small!")

print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

issues = []
if mu_variance < 0.1:
    issues.append(f"Mu variance is very low ({mu_variance:.6f}) - posterior collapse")
if qed_pred_np.std() < 0.01:
    issues.append(f"Property predictions have no variance (std={qed_pred_np.std():.6f})")
if weight_magnitude < 0.1:
    issues.append(f"Property head weights are very small (magnitude={weight_magnitude:.6f})")
if kl_loss_mean < 0.01:
    issues.append(f"KL loss is very low ({kl_loss_mean:.6f}) - latent space not being used")
if correlation < 0.5:
    issues.append(f"Property correlation is low ({correlation:.6f})")

if issues:
    print("\n⚠ Issues found:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✓ No obvious issues found")

print()

