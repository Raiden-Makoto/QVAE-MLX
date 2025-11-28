import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors

from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.reparameterize import Reparameterize
from model.vae import VAE
from utils.loss import compute_loss, categorical_crossentropy
import mlx.nn as nn
import mlx.core as mx
from train import load_data, create_model, validate


def compute_loss_components(
    z_logvar, z_mean, qed_true, qed_pred, graph_real, graph_generated,
    kl_weight=1.0, adj_weight=1.0, feat_weight=1.0, prop_weight=1.0, graph_weight=0.1
):
    """Compute individual loss components for display."""
    adjacency_real, features_real = graph_real
    adjacency_gen, features_gen = graph_generated
    
    # KL divergence loss
    kl_loss = -0.5 * mx.sum(1 + z_logvar - z_mean**2 - mx.exp(z_logvar), axis=1)
    kl_loss = mx.mean(kl_loss)
    kl_loss = mx.where(mx.isfinite(kl_loss), kl_loss, mx.zeros_like(kl_loss))
    
    # Property loss: binary crossentropy with probabilities
    qed_pred_squeezed = mx.squeeze(qed_pred, axis=1)
    epsilon = 1e-8
    property_loss = mx.mean(
        -(qed_true * mx.log(qed_pred_squeezed + epsilon) + 
          (1.0 - qed_true) * mx.log(1.0 - qed_pred_squeezed + epsilon))
    )
    property_loss = mx.where(mx.isfinite(property_loss), property_loss, mx.zeros_like(property_loss))
    
    # Reconstruction loss (adjacency + features)
    adjacency_gen_safe = mx.where(mx.isfinite(adjacency_gen), adjacency_gen, mx.zeros_like(adjacency_gen))
    adjacency_loss_per_sample = categorical_crossentropy(adjacency_real, adjacency_gen_safe, axis=1)
    adjacency_loss_summed = mx.sum(adjacency_loss_per_sample, axis=1)
    adjacency_loss_summed = mx.sum(adjacency_loss_summed, axis=1)
    num_atoms = adjacency_real.shape[2]
    norm_factor = mx.maximum(num_atoms * num_atoms, 1.0)
    adjacency_loss = mx.mean(adjacency_loss_summed) / norm_factor
    adjacency_loss = mx.where(mx.isfinite(adjacency_loss), adjacency_loss, mx.zeros_like(adjacency_loss))
    
    features_gen_safe = mx.where(mx.isfinite(features_gen), features_gen, mx.zeros_like(features_gen))
    features_loss_per_sample = categorical_crossentropy(features_real, features_gen_safe, axis=2)
    norm_factor = mx.maximum(num_atoms, 1.0)
    features_loss = mx.mean(mx.sum(features_loss_per_sample, axis=1)) / norm_factor
    features_loss = mx.where(mx.isfinite(features_loss), features_loss, mx.zeros_like(features_loss))
    
    recon_loss = adj_weight * adjacency_loss + feat_weight * features_loss
    
    total_loss = (
        kl_weight * kl_loss +
        prop_weight * property_loss +
        recon_loss
    )
    
    return {
        'kl': float(kl_loss),
        'property': float(property_loss),
        'recon': float(recon_loss),
        'total': float(total_loss)
    }


def evaluate_model(
    model,
    adjacency,
    features,
    qed_true,
    batch_size=20,
    num_samples=1000
):
    """Comprehensive evaluation of the model."""
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)
    
    model.eval()
    
    dataset_size = len(adjacency)
    num_eval_samples = min(num_samples, dataset_size)
    
    print(f"\nEvaluating on {num_eval_samples} samples...")
    
    # Metrics
    total_loss = 0.0
    total_kl_loss = 0.0
    total_adj_loss = 0.0
    total_feat_loss = 0.0
    total_prop_loss = 0.0
    total_graph_loss = 0.0
    
    # Property prediction metrics
    qed_predictions = []
    qed_targets = []
    
    num_batches = 0
    
    eval_range = range(0, num_eval_samples, batch_size)
    eval_pbar = tqdm(eval_range, desc="Evaluation", position=0, leave=True, bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}', unit='', unit_scale=False)
    
    for i in eval_pbar:
        end_idx = min(i + batch_size, num_eval_samples)
        adj_batch = adjacency[i:end_idx]
        feat_batch = features[i:end_idx]
        qed_batch = qed_true[i:end_idx]
        
        # Forward pass
        mu, logvar, adj_gen, feat_gen, qed_pred = model((adj_batch, feat_batch))
        
        # Compute loss components
        loss_components = compute_loss_components(
            logvar, mu, qed_batch, qed_pred,
            (adj_batch, feat_batch),
            (adj_gen, feat_gen),
            kl_weight=5.0,
            adj_weight=0.75,
            feat_weight=6.0,
            prop_weight=6.0,
            graph_weight=1.0
        )
        
        loss_val = loss_components['total']
        total_loss += loss_val
        total_kl_loss += loss_components['kl']
        total_prop_loss += loss_components['property']
        total_adj_loss += loss_components['recon']
        
        # Collect property predictions
        qed_pred_vals = np.array(qed_pred).flatten()
        qed_true_vals = np.array(qed_batch).flatten()
        qed_predictions.extend(qed_pred_vals)
        qed_targets.extend(qed_true_vals)
        
        num_batches += 1
        
        # Update progress bar with individual components
        eval_pbar.set_postfix({
            "loss": f"{loss_val:.4f}",
            "kl": f"{loss_components['kl']:.4f}",
            "prop": f"{loss_components['property']:.4f}",
            "recon": f"{loss_components['recon']:.4f}"
        })
    
    model.train()
    
    # Compute average metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Property prediction metrics
    qed_predictions = np.array(qed_predictions)
    qed_targets = np.array(qed_targets)
    qed_mae = np.mean(np.abs(qed_predictions - qed_targets))
    qed_mse = np.mean((qed_predictions - qed_targets) ** 2)
    qed_rmse = np.sqrt(qed_mse)
    
    # Correlation
    qed_correlation = np.corrcoef(qed_predictions, qed_targets)[0, 1]
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\nOverall Metrics:")
    print(f"  Average Loss: {avg_loss:.6f}")
    print(f"\nProperty Prediction (QED):")
    print(f"  MAE:  {qed_mae:.6f}")
    print(f"  MSE:  {qed_mse:.6f}")
    print(f"  RMSE: {qed_rmse:.6f}")
    print(f"  Correlation: {qed_correlation:.6f}")
    print(f"  Target range: [{qed_targets.min():.4f}, {qed_targets.max():.4f}]")
    print(f"  Pred range:   [{qed_predictions.min():.4f}, {qed_predictions.max():.4f}]")
    
    return {
        'avg_loss': avg_loss,
        'qed_mae': qed_mae,
        'qed_mse': qed_mse,
        'qed_rmse': qed_rmse,
        'qed_correlation': qed_correlation
    }


def generate_molecules(model, num_samples=10, batch_size=5):
    """Generate new molecules using the VAE inference method."""
    print("\n" + "=" * 70)
    print("MOLECULE GENERATION")
    print("=" * 70)
    
    print(f"\nGenerating {num_samples} molecules...")
    
    molecules = []
    valid_molecules = []
    
    # Generate in batches
    for i in range(0, num_samples, batch_size):
        batch_size_actual = min(batch_size, num_samples - i)
        mols = model.inference(batch_size_actual)
        molecules.extend(mols)
        
        # Count valid molecules
        for mol in mols:
            if mol is not None:
                valid_molecules.append(mol)
    
    print(f"\nGenerated {len(molecules)} molecules")
    print(f"Valid molecules: {len(valid_molecules)} ({len(valid_molecules)/len(molecules)*100:.1f}%)")
    if len(valid_molecules) == 0:
        print(f"  All molecules failed sanitization - model may need more training")
        print(f"  This is common for VAEs and indicates the decoder needs improvement")
    
    # Analyze valid molecules
    if valid_molecules:
        print(f"\nValid Molecule Statistics:")
        num_atoms_list = [mol.GetNumAtoms() for mol in valid_molecules]
        num_bonds_list = [mol.GetNumBonds() for mol in valid_molecules]
        qed_list = [Descriptors.qed(mol) for mol in valid_molecules]
        
        print(f"  Num atoms: {np.mean(num_atoms_list):.1f} ± {np.std(num_atoms_list):.1f} (range: {min(num_atoms_list)}-{max(num_atoms_list)})")
        print(f"  Num bonds: {np.mean(num_bonds_list):.1f} ± {np.std(num_bonds_list):.1f} (range: {min(num_bonds_list)}-{max(num_bonds_list)})")
        print(f"  QED: {np.mean(qed_list):.4f} ± {np.std(qed_list):.4f} (range: {np.min(qed_list):.4f}-{np.max(qed_list):.4f})")
        
        # Show first few valid molecules
        print(f"\nFirst 5 valid molecules (SMILES):")
        for i, mol in enumerate(valid_molecules[:5]):
            smiles = Chem.MolToSmiles(mol)
            qed = Descriptors.qed(mol)
            print(f"  {i+1}. {smiles} (QED: {qed:.4f}, Atoms: {mol.GetNumAtoms()})")
    
    return molecules, valid_molecules


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE model")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.npz", help="Checkpoint file to load")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for evaluation")
    parser.add_argument("--num_eval_samples", type=int, default=1000, help="Number of samples for evaluation")
    parser.add_argument("--num_generate", type=int, default=20, help="Number of molecules to generate")
    parser.add_argument("--latent_dim", type=int, default=435, help="Latent dimension")
    parser.add_argument("--no_generate", action="store_true", help="Skip molecule generation")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading dataset...")
    adjacency_tensor, feature_tensor, qed_tensor = load_data(args.data_dir)
    
    # Get shapes
    batch_size_total, bond_dim, num_atoms, _ = adjacency_tensor.shape
    _, _, atom_dim = feature_tensor.shape
    adjacency_shape = (bond_dim, num_atoms, num_atoms)
    feature_shape = (num_atoms, atom_dim)
    
    print(f"  Dataset size: {batch_size_total}")
    print(f"  Adjacency shape: {adjacency_shape}")
    print(f"  Feature shape: {feature_shape}")
    
    # Create model
    print(f"\nCreating model...")
    vae = create_model(args.latent_dim, adjacency_shape, feature_shape)
    
    # Load checkpoint
    checkpoint_path = f"{args.checkpoint_dir}/{args.checkpoint}"
    checkpoint_dir = Path(args.checkpoint_dir)
    
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Error: Checkpoint not found at {checkpoint_path}")
        
        if not checkpoint_dir.exists():
            print(f"  Checkpoint directory '{args.checkpoint_dir}' does not exist.")
            print(f"  Please train the model first using: python train.py")
        else:
            print(f"  Available checkpoints in '{args.checkpoint_dir}':")
            checkpoints = list(checkpoint_dir.glob("*.npz"))
            if checkpoints:
                for cp in sorted(checkpoints):
                    size_mb = cp.stat().st_size / (1024 * 1024)
                    print(f"    - {cp.name} ({size_mb:.1f} MB)")
            else:
                print(f"    (no checkpoints found)")
            print(f"\n  Please train the model first using: python train.py")
        return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    try:
        vae.load_weights(checkpoint_path)
        print(f"  ✓ Checkpoint loaded")
    except ValueError as e:
        # Handle shape mismatches (e.g., quantum layer weights when latent_dim changed)
        error_msg = str(e)
        if "q_dense.weights" in error_msg or "encoder.q_dense" in error_msg:
            print(f"  ⚠ Warning: Quantum layer weight mismatch (likely due to latent_dim change)")
            print(f"  Skipping quantum layer weights (layer is not used in forward pass anyway)")
            # Load weights manually, skipping the quantum layer
            import numpy as np
            weights_dict = np.load(checkpoint_path, allow_pickle=True)
            model_params = vae.parameters()
            
            def load_weights_skip_quantum(model_dict, weight_dict, prefix=""):
                loaded = 0
                skipped = 0
                for key, value in weight_dict.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    # Skip quantum layer weights
                    if "q_dense" in full_key:
                        print(f"    Skipping {full_key} (quantum layer - not used)")
                        skipped += 1
                        continue
                    
                    # Navigate to the right location in model_dict
                    keys = full_key.split('.')
                    current = model_dict
                    try:
                        for k in keys[:-1]:
                            current = current[k]
                        
                        target_key = keys[-1]
                        if target_key in current:
                            if hasattr(current[target_key], 'shape') and hasattr(value, 'shape'):
                                if current[target_key].shape == value.shape:
                                    current[target_key] = mx.array(value)
                                    loaded += 1
                                else:
                                    print(f"    Skipping {full_key}: shape mismatch")
                                    skipped += 1
                            else:
                                current[target_key] = mx.array(value) if not isinstance(value, mx.array) else value
                                loaded += 1
                    except (KeyError, TypeError) as ex:
                        print(f"    Skipping {full_key}: {ex}")
                        skipped += 1
                
                return loaded, skipped
            
            # Convert numpy arrays to dict structure
            weights_flat = {}
            for key in weights_dict.keys():
                weights_flat[key] = weights_dict[key]
            
            loaded_count, skipped_count = load_weights_skip_quantum(model_params, weights_flat)
            print(f"  ✓ Partially loaded: {loaded_count} parameters loaded, {skipped_count} skipped")
        else:
            raise e
    
    # Run evaluation
    metrics = evaluate_model(
        vae,
        adjacency_tensor,
        feature_tensor,
        qed_tensor,
        batch_size=args.batch_size,
        num_samples=args.num_eval_samples
    )
    
    # Generate molecules
    if not args.no_generate:
        molecules, valid_molecules = generate_molecules(
            vae,
            num_samples=args.num_generate,
            batch_size=5
        )
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

