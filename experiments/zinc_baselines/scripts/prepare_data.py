"""
Data preparation script for ZINC tranches.
Loads pre-split data from data/data_* folders and computes molecular properties.

Usage:
    python experiments/zinc_baselines/scripts/prepare_data.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from rdkit import Chem

# Import property computation from data folder
from data.property_utils import PROPERTIES_TO_COMPUTE, compute_props

# Configuration
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "zinc_baselines" / "data"

TRANCHES = ["bbab", "fbab", "jbcd"]  # lowercase folder names
PROPERTY_COLUMNS = PROPERTIES_TO_COMPUTE  # ['qed', 'HeavyAtomMolWt', 'MolWt', 'TPSA', 'MolLogP', 'num_atoms']


def load_smiles_from_csv(csv_path: Path) -> list:
    """Load SMILES from a CSV file."""
    df = pd.read_csv(csv_path)
    return df["smiles"].tolist()


def compute_properties_for_smiles(smiles_list: list) -> np.ndarray:
    """Compute molecular properties for a list of SMILES."""
    properties = []
    
    for smi in tqdm(smiles_list, desc="Computing properties", leave=False):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Use NaN for invalid molecules
            props = {p: float('nan') for p in PROPERTY_COLUMNS}
        else:
            props = compute_props(mol)
        
        # Extract properties in order
        prop_values = [props.get(p, float('nan')) for p in PROPERTY_COLUMNS]
        properties.append(prop_values)
    
    return np.array(properties, dtype=np.float32)


def load_and_prepare_tranche(tranche_name: str) -> dict:
    """Load a ZINC tranche and prepare for torch-molecule."""
    
    tranche_dir = DATA_DIR / f"data_{tranche_name}"
    
    if not tranche_dir.exists():
        raise FileNotFoundError(f"Tranche directory not found: {tranche_dir}")
    
    # Load SMILES from train/val/test splits
    train_csv = tranche_dir / "train" / "molecules.csv"
    val_csv = tranche_dir / "val" / "molecules.csv"
    test_csv = tranche_dir / "test" / "molecules.csv"
    
    print(f"  Loading train data from {train_csv}")
    X_train = load_smiles_from_csv(train_csv)
    print(f"  Loading val data from {val_csv}")
    X_val = load_smiles_from_csv(val_csv)
    print(f"  Loading test data from {test_csv}")
    X_test = load_smiles_from_csv(test_csv)
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Compute properties
    print(f"  Computing properties for train set...")
    y_train = compute_properties_for_smiles(X_train)
    print(f"  Computing properties for val set...")
    y_val = compute_properties_for_smiles(X_val)
    print(f"  Computing properties for test set...")
    y_test = compute_properties_for_smiles(X_test)
    
    # Compute statistics on training set (for normalization)
    # Handle NaN values
    y_train_clean = np.nan_to_num(y_train, nan=0.0)
    property_means = np.nanmean(y_train, axis=0).astype(np.float32)
    property_stds = np.nanstd(y_train, axis=0).astype(np.float32)
    
    # Replace zero std with 1 to avoid division by zero
    property_stds[property_stds == 0] = 1.0
    
    print(f"  Property statistics (mean ± std):")
    for i, prop in enumerate(PROPERTY_COLUMNS):
        print(f"    {prop}: {property_means[i]:.3f} ± {property_stds[i]:.3f}")
    
    data = {
        "tranche": tranche_name.upper(),
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "property_names": PROPERTY_COLUMNS,
        "property_means": property_means,
        "property_stds": property_stds,
    }
    
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for tranche in TRANCHES:
        print(f"\n{'='*60}")
        print(f"Processing {tranche.upper()}...")
        print(f"{'='*60}")
        
        try:
            data = load_and_prepare_tranche(tranche)
            
            # Save as pickle for easy loading
            output_path = OUTPUT_DIR / f"{tranche.upper()}_prepared.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
            print(f"  Saved to {output_path}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"{'='*60}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
