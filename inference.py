import mlx.core as mx
import mlx.nn as nn
import numpy as np
import argparse
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.reparameterize import Reparameterize
from model.vae import VAE
from train import load_data, create_model


def generate_molecules(model, num_samples=20, batch_size=5):
    """Generate new molecules using the VAE inference method."""
    print("=" * 70)
    print("MOLECULE GENERATION")
    print("=" * 70)
    
    print(f"\nGenerating {num_samples} molecules...")
    
    molecules = []
    valid_molecules = []
    
    # Generate in batches
    for i in range(0, num_samples, batch_size):
        batch_size_actual = min(batch_size, num_samples - i)
        print(f"  Generating batch {i//batch_size + 1} ({batch_size_actual} molecules)...")
        mols = model.inference(batch_size_actual)
        molecules.extend(mols)
        
        # Count valid molecules
        for mol in mols:
            if mol is not None:
                valid_molecules.append(mol)
    
    print(f"\nGenerated {len(molecules)} molecules")
    print(f"Valid molecules: {len(valid_molecules)} ({len(valid_molecules)/len(molecules)*100:.1f}%)")
    
    # Analyze valid molecules
    if valid_molecules:
        print(f"\nValid Molecule Statistics:")
        num_atoms_list = [mol.GetNumAtoms() for mol in valid_molecules]
        num_bonds_list = [mol.GetNumBonds() for mol in valid_molecules]
        qed_list = [Descriptors.qed(mol) for mol in valid_molecules]
        
        print(f"  Num atoms: {np.mean(num_atoms_list):.1f} ± {np.std(num_atoms_list):.1f} (range: {min(num_atoms_list)}-{max(num_atoms_list)})")
        print(f"  Num bonds: {np.mean(num_bonds_list):.1f} ± {np.std(num_bonds_list):.1f} (range: {min(num_bonds_list)}-{max(num_bonds_list)})")
        print(f"  QED: {np.mean(qed_list):.4f} ± {np.std(qed_list):.4f} (range: {np.min(qed_list):.4f}-{np.max(qed_list):.4f})")
        
        # Show all valid molecules
        print(f"\nAll valid molecules (SMILES):")
        for i, mol in enumerate(valid_molecules):
            smiles = Chem.MolToSmiles(mol)
            qed = Descriptors.qed(mol)
            num_atoms = mol.GetNumAtoms()
            num_bonds = mol.GetNumBonds()
            print(f"  {i+1}. {smiles}")
            print(f"      QED: {qed:.4f}, Atoms: {num_atoms}, Bonds: {num_bonds}")
    else:
        print(f"\n  No valid molecules generated - model may need more training")
        print(f"  This is common for VAEs and indicates the decoder needs improvement")
    
    return molecules, valid_molecules


def main():
    parser = argparse.ArgumentParser(description="Generate molecules using VAE")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.npz", help="Checkpoint file to load")
    parser.add_argument("--num_generate", type=int, default=20, help="Number of molecules to generate")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for generation")
    parser.add_argument("--latent_dim", type=int, default=435, help="Latent dimension")
    
    args = parser.parse_args()
    
    # Load data to get shapes
    print("Loading dataset to get shapes...")
    adjacency_tensor, feature_tensor, _ = load_data("data")
    
    # Get shapes
    batch_size_total, bond_dim, num_atoms, _ = adjacency_tensor.shape
    _, _, atom_dim = feature_tensor.shape
    adjacency_shape = (bond_dim, num_atoms, num_atoms)
    feature_shape = (num_atoms, atom_dim)
    
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
    
    # Generate molecules
    molecules, valid_molecules = generate_molecules(
        vae,
        num_samples=args.num_generate,
        batch_size=args.batch_size
    )
    
    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

