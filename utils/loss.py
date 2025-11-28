import mlx.core as mx
import mlx.nn as nn


def categorical_crossentropy(y_true, y_pred, axis=None):
    """
    Categorical crossentropy loss.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        axis: Axis along which to compute (if None, computes over all)
    
    Returns:
        Loss value
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-8
    # Categorical crossentropy: -sum(y_true * log(y_pred))
    # Add small epsilon to prevent log(0) = -inf
    eps = 1e-8
    # Use y_pred directly without clipping
    y_pred_safe = y_pred
    # Normalize to ensure it sums to 1 (softmax-like behavior)
    if axis is not None:
        y_pred_safe = y_pred_safe / (mx.sum(y_pred_safe, axis=axis, keepdims=True) + eps)
    if axis is None:
        loss = -mx.sum(y_true * mx.log(y_pred_safe + eps))
    else:
        loss = -mx.sum(y_true * mx.log(y_pred_safe + eps), axis=axis)
    # Ensure loss is finite
    loss = mx.where(mx.isfinite(loss), loss, mx.zeros_like(loss))
    return loss


def compute_loss(
    z_logvar, z_mean, qed_true, qed_pred, graph_real, graph_generated,
    kl_weight=1.0, adj_weight=1.0, feat_weight=1.0, prop_weight=1.0, graph_weight=0.1
):
    """
    Compute VAE loss with component weighting.
    
    Args:
        z_logvar: Log variance of latent distribution (batch, latent_dim)
        z_mean: Mean of latent distribution (batch, latent_dim)
        qed_true: True QED property values (batch,)
        qed_pred: Predicted QED property values (batch, 1)
        graph_real: Tuple of (adjacency_real, features_real)
        graph_generated: Tuple of (adjacency_gen, features_gen)
        kl_weight: Weight for KL divergence loss
        adj_weight: Weight for adjacency loss
        feat_weight: Weight for features loss
        prop_weight: Weight for property loss
        graph_weight: Weight for gradient penalty
    
    Returns:
        Total loss
    """
    adjacency_real, features_real = graph_real
    adjacency_gen, features_gen = graph_generated
    
    # Adjacency loss: categorical crossentropy
    # Normalize by number of atoms to make it more comparable
    # Ensure inputs are finite
    adjacency_gen_safe = mx.where(mx.isfinite(adjacency_gen), adjacency_gen, mx.zeros_like(adjacency_gen))
    adjacency_loss_per_sample = categorical_crossentropy(
        adjacency_real, adjacency_gen_safe, axis=1
    )  # (batch, num_atoms, num_atoms)
    adjacency_loss_summed = mx.sum(adjacency_loss_per_sample, axis=1)  # (batch, num_atoms)
    adjacency_loss_summed = mx.sum(adjacency_loss_summed, axis=1)  # (batch,)
    num_atoms = adjacency_real.shape[2]
    # Avoid division by zero
    norm_factor = mx.maximum(num_atoms * num_atoms, 1.0)
    adjacency_loss = mx.mean(adjacency_loss_summed) / norm_factor  # Normalize
    # Ensure finite
    adjacency_loss = mx.where(mx.isfinite(adjacency_loss), adjacency_loss, mx.zeros_like(adjacency_loss))
    
    # Features loss: categorical crossentropy
    # Ensure inputs are finite
    features_gen_safe = mx.where(mx.isfinite(features_gen), features_gen, mx.zeros_like(features_gen))
    features_loss_per_sample = categorical_crossentropy(
        features_real, features_gen_safe, axis=2
    )  # (batch, num_atoms)
    # Avoid division by zero
    norm_factor = mx.maximum(num_atoms, 1.0)
    features_loss = mx.mean(mx.sum(features_loss_per_sample, axis=1)) / norm_factor  # Normalize
    # Ensure finite
    features_loss = mx.where(mx.isfinite(features_loss), features_loss, mx.zeros_like(features_loss))
    
    # KL divergence loss: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * mx.sum(
        1 + z_logvar - z_mean**2 - mx.exp(z_logvar),
        axis=1
    )
    kl_loss = mx.mean(kl_loss)
    # Ensure finite
    kl_loss = mx.where(mx.isfinite(kl_loss), kl_loss, mx.zeros_like(kl_loss))
    
    # NOTE: Removed minimum KL constraint - it was causing numerical issues
    # The real fix is to increase KL weight and improve encoder initialization
    
    # Property loss: binary crossentropy
    # qed_pred is now probabilities [0, 1] after sigmoid in VAE
    qed_pred_squeezed = mx.squeeze(qed_pred, axis=1)  # (batch,)
    # Add small epsilon for numerical stability in log
    epsilon = 1e-8
    # Binary crossentropy with probabilities: -[y*log(p) + (1-y)*log(1-p)]
    property_loss = mx.mean(
        -(qed_true * mx.log(qed_pred_squeezed + epsilon) + 
          (1.0 - qed_true) * mx.log(1.0 - qed_pred_squeezed + epsilon))
    )
    # Ensure finite
    property_loss = mx.where(mx.isfinite(property_loss), property_loss, mx.zeros_like(property_loss))
    
    # Graph loss (gradient penalty) - simplified for now
    graph_loss = gradient_penalty(graph_real, graph_generated)
    # Ensure finite
    graph_loss = mx.where(mx.isfinite(graph_loss), graph_loss, mx.zeros_like(graph_loss))
    
    # Weighted total loss
    total_loss = (
        kl_weight * kl_loss +
        prop_weight * property_loss +
        graph_weight * graph_loss +
        adj_weight * adjacency_loss +
        feat_weight * features_loss
    )
    
    # Final safety check - if total is NaN/Inf, return a large but finite value
    total_loss = mx.where(mx.isfinite(total_loss), total_loss, mx.array(1000.0))
    
    return total_loss


def gradient_penalty(graph_real, graph_generated, vae_model=None):
    """
    Compute gradient penalty for graph adversarial training.
    
    Note: This is a simplified version. Full implementation would require
    computing gradients with respect to interpolated graphs, which needs
    the VAE model to be passed in.
    
    Args:
        graph_real: Tuple of (adjacency_real, features_real)
        graph_generated: Tuple of (adjacency_generated, features_generated)
        vae_model: VAE model (optional, for full gradient penalty)
    
    Returns:
        Gradient penalty loss
    """
    adjacency_real, features_real = graph_real
    adjacency_generated, features_generated = graph_generated
    
    batch_size = adjacency_real.shape[0]
    
    # Generate interpolated graphs
    alpha = mx.random.uniform(0, 1, (batch_size,), dtype=mx.float32)
    alpha_adj = mx.reshape(alpha, (batch_size, 1, 1, 1))
    alpha_feat = mx.reshape(alpha, (batch_size, 1, 1))
    
    adjacency_interp = adjacency_real * alpha_adj + (1.0 - alpha_adj) * adjacency_generated
    features_interp = features_real * alpha_feat + (1.0 - alpha_feat) * features_generated
    
    # Simplified gradient penalty: compute norm of difference
    # Full implementation would compute gradients of discriminator/critic
    # For now, return a small regularization term
    adj_diff = adjacency_interp - adjacency_real
    feat_diff = features_interp - features_real
    
    # Compute norms
    adj_norm = mx.linalg.norm(adj_diff, axis=1)  # (batch, num_atoms, num_atoms)
    feat_norm = mx.linalg.norm(feat_diff, axis=2)  # (batch, num_atoms)
    
    # Penalty: (1 - norm)^2
    adj_penalty = mx.mean((1.0 - mx.mean(adj_norm, axis=(-2, -1))) ** 2)
    feat_penalty = mx.mean((1.0 - mx.mean(feat_norm, axis=-1)) ** 2)
    
    return adj_penalty + feat_penalty
