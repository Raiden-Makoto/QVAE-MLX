import mlx.core as mx
import numpy as np
import pandas as pd
import ast
import pickle
import os
import argparse

from rdkit import Chem, RDLogger
from rdkit.Chem import BondType

RDLogger.DisableLog('rdApp.*')

SMILE_CHARSET = '["C", "B", "F", "I", "H", "O", "N", "S", "P", "Cl", "Br"]'

bond_mapping = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
bond_mapping.update(
    {0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC}
)
SMILE_CHARSET = ast.literal_eval(SMILE_CHARSET)

csv_path = "data/250k_rndm_zinc_drugs_clean_3.csv"
df = pd.read_csv(csv_path)

MAX_MOLSIZE = max(df["smiles"].str.len())
SMILE_to_index = dict((c, i) for i, c in enumerate(SMILE_CHARSET))
index_to_SMILE = dict((i, c) for i, c in enumerate(SMILE_CHARSET))
atom_mapping = dict(SMILE_to_index)
atom_mapping.update(index_to_SMILE)

BATCH_SIZE = 100
NUM_ATOMS = 90

ATOM_DIM = len(SMILE_CHARSET)
BOND_DIM = 4 + 1
LATENT_DIM = 435


def smiles_to_graph(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    adjacency = mx.zeros((BOND_DIM, NUM_ATOMS, NUM_ATOMS), dtype=mx.float32)
    features = mx.zeros((NUM_ATOMS, ATOM_DIM), dtype=mx.float32)

    for atom in molecule.GetAtoms():
        i = atom.GetIdx()
        atom_type = atom_mapping[atom.GetSymbol()]
        one_hot = mx.zeros(ATOM_DIM, dtype=mx.float32)
        one_hot[atom_type] = 1.0
        features[i] = one_hot
        for neighbor in atom.GetNeighbors():
            j = neighbor.GetIdx()
            bond = molecule.GetBondBetweenAtoms(i, j)
            bond_type_idx = bond_mapping[bond.GetBondType().name]
            adjacency[bond_type_idx, i, j] = 1.0
            adjacency[bond_type_idx, j, i] = 1.0

    bond_sum = mx.sum(adjacency[:-1], axis=0)
    no_bond_mask = bond_sum == 0
    adjacency_last = mx.where(no_bond_mask, 1.0, adjacency[-1])
    adjacency = mx.concatenate([adjacency[:-1], adjacency_last[None]], axis=0)

    feature_sum = mx.sum(features, axis=1)
    no_atom_mask = feature_sum == 0
    features_last_col = mx.where(no_atom_mask[:, None], 1.0, features[:, -1:])
    features = mx.concatenate([features[:, :-1], features_last_col], axis=1)

    return adjacency, features


def graph_to_molecule(graph):
    adjacency, features = graph

    molecule = Chem.RWMol()

    atom_types = mx.argmax(features, axis=1)
    atom_mask = atom_types != ATOM_DIM - 1
    
    bond_sum_per_atom = mx.sum(adjacency[:-1], axis=0)
    bond_sum_per_atom = mx.sum(bond_sum_per_atom, axis=1)
    bond_mask = bond_sum_per_atom > 0
    
    keep_mask = atom_mask
    
    keep_idx = [i for i in range(len(keep_mask)) if bool(keep_mask[i])]
    
    if len(keep_idx) == 0:
        return None
    
    features = features[keep_idx]
    adjacency = adjacency[:, keep_idx, :][:, :, keep_idx]

    atom_types = mx.argmax(features, axis=1)
    for atom_type_idx in atom_types:
        atom = Chem.Atom(atom_mapping[int(atom_type_idx)])
        _ = molecule.AddAtom(atom)

    num_atoms = features.shape[0]
    triu_mask = mx.zeros((num_atoms, num_atoms), dtype=mx.bool_)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            triu_mask[i, j] = True
    
    added_bonds = set()
    for bond_idx in range(adjacency.shape[0]):
        if bond_idx == BOND_DIM - 1:
            continue
        bond_matrix = adjacency[bond_idx]
        bond_exists = (bond_matrix == 1.0) & triu_mask
        bond_locations = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if bool(bond_exists[i, j]):
                    bond_pair = (min(i, j), max(i, j))
                    if bond_pair not in added_bonds:
                        bond_locations.append((i, j))
                        added_bonds.add(bond_pair)
        
        for atom_i, atom_j in bond_locations:
            bond_type = bond_mapping[bond_idx]
            try:
                molecule.AddBond(int(atom_i), int(atom_j), bond_type)
            except:
                pass
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        return None

    return molecule

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess molecular dataset")
    parser.add_argument("--percent", type=float, default=10.0, 
                       help="Percentage of dataset to load (default: 10.0)")
    args = parser.parse_args()
    
    df = pd.read_csv(csv_path)
    train_df = df.sample(frac=0.75, random_state=67)
    train_df.reset_index(drop=True, inplace=True)
    
    # Calculate number of samples based on percentage
    total_available = len(train_df)
    num_samples = int(total_available * (args.percent / 100.0))
    
    print(f"Dataset preprocessing:")
    print(f"  Total available samples: {total_available:,}")
    print(f"  Loading {args.percent}% of dataset: {num_samples:,} samples")
    
    adjacency_list, feature_list, qed_list = [], [], []
    
    for idx in range(num_samples):
        if idx % 1000 == 0:
            print(f"Processing {idx}/{num_samples}...")
        adjacency, features = smiles_to_graph(train_df.loc[idx]["smiles"])
        qed = train_df.loc[idx]["qed"]
        adjacency_list.append(np.array(adjacency))
        feature_list.append(np.array(features))
        qed_list.append(qed)
    
    print("Converting to numpy arrays...")
    adjacency_tensor = np.array(adjacency_list)
    feature_tensor = np.array(feature_list)
    qed_tensor = np.array(qed_list)
    
    print("Converting to MLX arrays...")
    adjacency_tensor = mx.array(adjacency_tensor)
    feature_tensor = mx.array(feature_tensor)
    qed_tensor = mx.array(qed_tensor)

    os.makedirs("data", exist_ok=True)
    
    with open("data/adjacency_tensor.pkl", "wb") as f:
        pickle.dump(adjacency_tensor, f)
    with open("data/feature_tensor.pkl", "wb") as f:
        pickle.dump(feature_tensor, f)
    with open("data/qed_tensor.pkl", "wb") as f:
        pickle.dump(qed_tensor, f)
    
    print(f"Saved tensors to data folder:")
    print(f"  - adjacency_tensor: {adjacency_tensor.shape}")
    print(f"  - feature_tensor: {feature_tensor.shape}")
    print(f"  - qed_tensor: {qed_tensor.shape}")