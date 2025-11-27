import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pickle
import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm
import psutil
import gc

from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.reparameterize import Reparameterize
from model.vae import VAE
from utils.loss import compute_loss


def load_data(data_dir="data"):
    """Load preprocessed dataset."""
    print("Loading dataset...")
    with open(f"{data_dir}/adjacency_tensor.pkl", "rb") as f:
        adjacency_tensor = pickle.load(f)
    with open(f"{data_dir}/feature_tensor.pkl", "rb") as f:
        feature_tensor = pickle.load(f)
    with open(f"{data_dir}/qed_tensor.pkl", "rb") as f:
        qed_tensor = pickle.load(f)
    
    # Convert to MLX if needed
    if isinstance(adjacency_tensor, np.ndarray):
        adjacency_tensor = mx.array(adjacency_tensor)
    if isinstance(feature_tensor, np.ndarray):
        feature_tensor = mx.array(feature_tensor)
    if isinstance(qed_tensor, np.ndarray):
        qed_tensor = mx.array(qed_tensor)
    
    print(f"  Loaded {len(adjacency_tensor)} samples")
    return adjacency_tensor, feature_tensor, qed_tensor


def create_model(latent_dim, adjacency_shape, feature_shape):
    """Create and initialize VAE model."""
    encoder = Encoder(
        gconv_units=[512],
        latent_dim=latent_dim,
        adjacency_shape=adjacency_shape,
        feature_shape=feature_shape,
        dense_units=[512],
        dropout_rate=0.0
    )
    
    decoder = Decoder(
        latent_dim=latent_dim,
        dense_units=[128, 256, 512],
        dropout_rate=0.2,
        adjacency_shape=adjacency_shape,
        feature_shape=feature_shape
    )
    
    reparameterize = Reparameterize()
    vae = VAE(encoder, decoder, reparameterize)
    
    # Store shapes for inference
    encoder.latent_dim = latent_dim
    encoder.adjacency_shape = adjacency_shape
    encoder.feature_shape = feature_shape
    
    return vae


def make_loss_fn(model, kl_weight=1.0, adj_weight=1.0, feat_weight=1.0, prop_weight=1.0, graph_weight=0.1):
    """Create a loss function closure that captures the model."""
    def loss_fn(adjacency, features, qed_true):
        """Compute loss for a batch."""
        mu, logvar, adj_gen, feat_gen, qed_pred = model((adjacency, features))
        
        graph_real = (adjacency, features)
        graph_generated = (adj_gen, feat_gen)
        
        loss = compute_loss(
            logvar, mu, qed_true, qed_pred, graph_real, graph_generated,
            kl_weight=kl_weight, adj_weight=adj_weight, feat_weight=feat_weight,
            prop_weight=prop_weight, graph_weight=graph_weight
        )
        return loss
    return loss_fn


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB


def train_step(model, optimizer, loss_and_grad_fn, adjacency, features, qed_true, max_grad_norm=1.0):
    """Perform one training step."""
    # loss_and_grad_fn is created with nn.value_and_grad(model, loss_fn)
    # so it takes (adjacency, features, qed_true) as arguments
    loss, grads = loss_and_grad_fn(adjacency, features, qed_true)
    
    # Clip gradients to prevent explosion
    if max_grad_norm > 0:
        grads, _ = optim.clip_grad_norm(grads, max_norm=max_grad_norm)
    
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state)
    return loss


def validate(model, adjacency, features, qed_true, batch_size=20, num_samples=None):
    """Validate model on a subset of data."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    dataset_size = len(adjacency)
    if num_samples is None:
        num_val_samples = min(500, dataset_size)  # Validate on up to 500 samples
    else:
        num_val_samples = min(num_samples, dataset_size)
    
    val_range = range(0, num_val_samples, batch_size)
    val_pbar = tqdm(val_range, desc="Validation", position=1, leave=False)
    
    for i in val_pbar:
        end_idx = min(i + batch_size, num_val_samples)
        adj_batch = adjacency[i:end_idx]
        feat_batch = features[i:end_idx]
        qed_batch = qed_true[i:end_idx]
        
        mu, logvar, adj_gen, feat_gen, qed_pred = model((adj_batch, feat_batch))
        loss = compute_loss(
            logvar, mu, qed_batch, qed_pred,
            (adj_batch, feat_batch),
            (adj_gen, feat_gen)
        )
        
        loss_val = float(loss)
        total_loss += loss_val
        num_batches += 1
        
        # Update progress bar
        val_pbar.set_postfix({"loss": f"{loss_val:.4f}"})
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_epoch_{epoch}.npz"
    
    # Save model parameters
    model.save_weights(checkpoint_path)
    
    # Save optimizer state separately
    opt_path = f"{checkpoint_dir}/optimizer_epoch_{epoch}.npz"
    # Note: MLX optimizers don't have direct save, so we'll save epoch and loss
    np.savez(opt_path, epoch=epoch, loss=loss)
    
    print(f"  Saved checkpoint: {checkpoint_path}")


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """Load model checkpoint."""
    if os.path.exists(checkpoint_path):
        model.load_weights(checkpoint_path)
        print(f"  Loaded checkpoint: {checkpoint_path}")
        return True
    return False


def train(
    data_dir="data",
    batch_size=20,
    num_epochs=100,
    learning_rate=0.001,
    latent_dim=9,
    checkpoint_dir="checkpoints",
    resume_from=None,
    val_interval=5,
    save_interval=10,
    max_grad_norm=5.0,  # Increase from 1.0 to allow larger gradients
    early_stopping_patience=10,  # Stop if no improvement for N epochs
    early_stopping_min_delta=0.001,  # Minimum change to qualify as improvement
    kl_weight=1.0,
    adj_weight=0.15,  # Increased from 0.05 to give adjacency more importance
    feat_weight=1.2,
    prop_weight=1.2,
    graph_weight=0.2
):
    """Main training loop."""
    print("=" * 70)
    print("VAE TRAINING")
    print("=" * 70)
    
    # Load data
    adjacency_tensor, feature_tensor, qed_tensor = load_data(data_dir)
    
    # Get shapes
    batch_size_total, bond_dim, num_atoms, _ = adjacency_tensor.shape
    _, _, atom_dim = feature_tensor.shape
    adjacency_shape = (bond_dim, num_atoms, num_atoms)
    feature_shape = (num_atoms, atom_dim)
    
    print(f"\nDataset info:")
    print(f"  Total samples: {batch_size_total}")
    print(f"  Adjacency shape: {adjacency_shape}")
    print(f"  Feature shape: {feature_shape}")
    print(f"\nLoss weights: KL={kl_weight}, Adj={adj_weight}, Feat={feat_weight}, Prop={prop_weight}, Graph={graph_weight}")
    
    # Create model
    print(f"\nCreating model...")
    vae = create_model(latent_dim, adjacency_shape, feature_shape)
    print(f"  ✓ Model created")
    
    # Create optimizer with learning rate scheduling
    # Start with moderate LR increase to avoid NaN
    # With larger model (512 units), use more conservative LR
    initial_lr = learning_rate * 2  # Reduced from 3x to 2x for stability with larger model
    # Create scheduler that decays over total training steps (epochs * batches per epoch)
    total_steps = num_epochs * ((batch_size_total + batch_size - 1) // batch_size)
    scheduler = optim.schedulers.cosine_decay(initial_lr, total_steps, end=learning_rate)
    optimizer = optim.Adam(learning_rate=scheduler)
    print(f"  ✓ Optimizer created (Adam, initial_lr={initial_lr:.6f}, cosine decay to {learning_rate:.6f})")
    
    # Create loss and gradient function with balanced weights
    loss_fn = make_loss_fn(
        vae,
        kl_weight=kl_weight,
        adj_weight=adj_weight,
        feat_weight=feat_weight,
        prop_weight=prop_weight,
        graph_weight=graph_weight
    )
    loss_and_grad_fn = nn.value_and_grad(vae, loss_fn)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from:
        if load_checkpoint(vae, resume_from, optimizer):
            # Try to load epoch number
            opt_path = resume_from.replace("checkpoint_", "optimizer_").replace(".npz", ".npz")
            if os.path.exists(opt_path):
                opt_data = np.load(opt_path, allow_pickle=True)
                start_epoch = int(opt_data['epoch']) + 1
                print(f"  Resuming from epoch {start_epoch}")
    
    # Training loop
    print(f"\nStarting training...")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Total batches per epoch: {batch_size_total // batch_size}")
    
    # Initial memory usage
    initial_memory = get_memory_usage()
    print(f"  Initial memory usage: {initial_memory:.1f} MB")
    
    num_batches_per_epoch = (batch_size_total + batch_size - 1) // batch_size
    
    # Track memory usage
    memory_samples = []
    
    # Early stopping tracking
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0
    
    # Outer progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Epochs", position=0, leave=True)
    
    for epoch in epoch_pbar:
        epoch_start_time = time.time()
        epoch_losses = []
        epoch_memory_samples = []
        
        # Learning rate is automatically updated by scheduler in optimizer
        # Get current LR for display
        current_lr = float(optimizer.learning_rate) if hasattr(optimizer.learning_rate, 'item') else float(optimizer.learning_rate)
        
        # Shuffle data (simple approach: random indices)
        indices = np.random.permutation(batch_size_total)
        
        # Inner progress bar for batches
        batch_range = range(0, batch_size_total, batch_size)
        batch_pbar = tqdm(batch_range, desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
        
        for batch_idx in batch_pbar:
            end_idx = min(batch_idx + batch_size, batch_size_total)
            batch_indices = indices[batch_idx:end_idx]
            
            # Get batch - convert indices to list for MLX indexing
            adj_batch = adjacency_tensor[batch_indices.tolist()]
            feat_batch = feature_tensor[batch_indices.tolist()]
            qed_batch = qed_tensor[batch_indices.tolist()]
            
            # Training step with gradient clipping
            loss = train_step(vae, optimizer, loss_and_grad_fn, adj_batch, feat_batch, qed_batch, max_grad_norm=max_grad_norm)
            loss_val = float(loss)
            
            # Check for NaN and skip if found
            if np.isnan(loss_val) or not np.isfinite(loss_val):
                epoch_pbar.write(f"  ⚠ NaN loss detected at batch {batch_idx//batch_size + 1}, skipping...")
                continue
            
            epoch_losses.append(loss_val)
            
            # Monitor memory usage periodically
            if batch_idx % (batch_size * 10) == 0:  # Every 10 batches
                current_memory = get_memory_usage()
                epoch_memory_samples.append(current_memory)
                memory_samples.append(current_memory)
            
            # Update progress bar with current loss and memory
            if len(epoch_losses) > 0:
                recent_losses = epoch_losses[-10:] if len(epoch_losses) >= 10 else epoch_losses
                avg_loss = np.mean(recent_losses)
                current_memory = get_memory_usage()
                batch_pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "mem": f"{current_memory:.0f}MB"
                })
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = np.mean(epoch_losses)
        
        # Memory statistics for this epoch
        if epoch_memory_samples:
            avg_memory = np.mean(epoch_memory_samples)
            max_memory = np.max(epoch_memory_samples)
            min_memory = np.min(epoch_memory_samples)
        else:
            avg_memory = get_memory_usage()
            max_memory = avg_memory
            min_memory = avg_memory
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            "loss": f"{avg_epoch_loss:.4f}",
            "lr": f"{current_lr:.6f}",
            "time": f"{epoch_time:.1f}s",
            "samples/s": f"{batch_size_total/epoch_time:.1f}",
            "mem": f"{avg_memory:.0f}MB"
        })
        
        # Periodic garbage collection to help with memory
        if (epoch + 1) % 5 == 0:
            gc.collect()
            mx.clear_cache()
        
        # Validation
        if (epoch + 1) % val_interval == 0:
            val_loss = validate(vae, adjacency_tensor, feature_tensor, qed_tensor, batch_size=batch_size)
            epoch_pbar.write(f"  Validation loss: {val_loss:.4f} (lr={current_lr:.6f})")
            
            # Early stopping check
            if val_loss < best_val_loss - early_stopping_min_delta:
                # Improvement detected
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                
                # Save best model
                best_checkpoint_path = f"{checkpoint_dir}/checkpoint_best.npz"
                vae.save_weights(best_checkpoint_path)
                epoch_pbar.write(f"  ✓ New best validation loss: {best_val_loss:.4f} (saved to {best_checkpoint_path})")
            else:
                # No improvement - increment by val_interval since we only check every N epochs
                epochs_without_improvement += val_interval
                epoch_pbar.write(f"  No improvement for {epochs_without_improvement} epochs (best: {best_val_loss:.4f} at epoch {best_epoch})")
                
                # Check if we should stop
                if epochs_without_improvement >= early_stopping_patience:
                    epoch_pbar.write(f"\n  ⚠ Early stopping triggered!")
                    epoch_pbar.write(f"  Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                    epoch_pbar.write(f"  Stopping training at epoch {epoch + 1}")
                    break
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(vae, optimizer, epoch + 1, avg_epoch_loss, checkpoint_dir)
        
        print()
    
    # Save final checkpoint
    print("Training complete!")
    
    # Load best model if it exists (from early stopping)
    best_checkpoint_path = f"{checkpoint_dir}/checkpoint_best.npz"
    if os.path.exists(best_checkpoint_path) and best_epoch > 0:
        vae.load_weights(best_checkpoint_path)
        print(f"  Loaded best model from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    
    save_checkpoint(vae, optimizer, num_epochs, avg_epoch_loss, checkpoint_dir)
    
    # Final memory statistics
    final_memory = get_memory_usage()
    if memory_samples:
        peak_memory = np.max(memory_samples)
        avg_memory = np.mean(memory_samples)
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage Summary:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Peak: {peak_memory:.1f} MB")
        print(f"  Average: {avg_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB ({memory_increase/initial_memory*100:.1f}%)")
        
        # Check for memory issues
        if peak_memory > initial_memory * 2:
            print(f"  ⚠ Warning: Peak memory is {peak_memory/initial_memory:.1f}x initial memory")
        if memory_increase > 1000:  # More than 1GB increase
            print(f"  ⚠ Warning: Memory increased by {memory_increase:.1f} MB")
    
    return vae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE model")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--latent_dim", type=int, default=9, help="Latent dimension")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval (epochs)")
    parser.add_argument("--save_interval", type=int, default=10, help="Save interval (epochs)")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001, help="Minimum change to qualify as improvement")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="Weight for KL divergence loss")
    parser.add_argument("--adj_weight", type=float, default=0.15, help="Weight for adjacency loss")
    parser.add_argument("--feat_weight", type=float, default=1.2, help="Weight for node feature loss")
    parser.add_argument("--prop_weight", type=float, default=1.2, help="Weight for property prediction loss")
    parser.add_argument("--graph_weight", type=float, default=0.2, help="Weight for gradient penalty loss")
    
    args = parser.parse_args()
    
    train(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        latent_dim=args.latent_dim,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        val_interval=args.val_interval,
        save_interval=args.save_interval,
        max_grad_norm=args.max_grad_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        kl_weight=args.kl_weight,
        adj_weight=args.adj_weight,
        feat_weight=args.feat_weight,
        prop_weight=args.prop_weight,
        graph_weight=args.graph_weight
    )

