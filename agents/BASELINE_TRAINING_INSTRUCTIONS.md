# Baseline Training Instructions: Graph DiT and DiGress

## Overview

This document provides step-by-step instructions for training **Graph DiT** and **DiGress** baseline models on ZINC tranches (BBAB, FBAB, JBCD) for comparison with SCULPT. The goal is to produce results for:

1. **Table 1**: Unconditional generation metrics (Validity, Uniqueness, Novelty, Diversity, Similarity, FCD)
2. **Table 2**: Multi-property conditional generation (conditioning on QED, HeavyAtomMolWt, MolWt, TPSA, MolLogP, num_atoms)

---

## Prerequisites

### System Requirements
- Python 3.9+ (tested with 3.11.7)
- CUDA-compatible GPU (A100/L40S/H200 recommended)
- At least 32GB GPU memory for larger tranches

### Data Available
You will be provided with ZINC tranches in CSV format with the following columns:
- `smiles`: SMILES string
- `QED`: Quantitative Estimate of Drug-likeness
- `HeavyAtomMolWt`: Heavy atom molecular weight
- `MolWt`: Molecular weight
- `TPSA`: Topological polar surface area
- `MolLogP`: Partition coefficient (logP)
- `num_atoms`: Number of atoms
- `split`: One of `train`, `val`, `test`

Data location will be specified when you begin.

---

## Step 1: Fork and Clone Repository

### 1.1 Fork torch-molecule
Go to https://github.com/liugangcode/torch-molecule and click "Fork" to create your own copy.

### 1.2 Clone Your Fork
```bash
git clone https://github.com/<YOUR_USERNAME>/torch-molecule.git
cd torch-molecule
```

### 1.3 Create Project Structure
```bash
mkdir -p experiments/zinc_baselines/{data,configs,scripts,results,checkpoints}
mkdir -p experiments/zinc_baselines/results/{graph_dit,digress}
```

---

## Step 2: Environment Setup

### 2.1 Create Conda Environment
```bash
conda create -n torch_molecule python=3.11.7 -y
conda activate torch_molecule
```

### 2.2 Install PyTorch (adjust CUDA version as needed)
```bash
# For CUDA 11.8
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2.3 Install torch-molecule
```bash
pip install -e .
```

### 2.4 Install Additional Dependencies
```bash
# PyTorch Geometric (adjust for your PyTorch/CUDA version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# Evaluation packages
pip install fcd_torch
pip install git+https://github.com/igor-krawczuk/mini-moses

# Additional utilities
pip install rdkit pandas numpy scikit-learn tqdm wandb hydra-core omegaconf pytorch-lightning
```

### 2.5 Verify Installation
```python
import torch_molecule
from torch_molecule import GraphDiTMolecularGenerator, DigressMolecularGenerator
print("torch-molecule version:", torch_molecule.__version__)
print("Installation successful!")
```

---

## Step 3: Data Preparation

### 3.1 Data Format
torch-molecule expects:
- `X_train`, `X_val`: Lists of SMILES strings
- `y_train`, `y_val`: NumPy arrays of shape `(n_samples, n_properties)` for conditional generation, or `None` for unconditional

### 3.2 Create Data Loading Script
Create `experiments/zinc_baselines/scripts/prepare_data.py`:

```python
"""
Data preparation script for ZINC tranches.
Converts CSV files to torch-molecule format.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Configuration - UPDATE THESE PATHS
DATA_DIR = Path("/path/to/your/zinc/data")  # <-- UPDATE THIS
OUTPUT_DIR = Path("experiments/zinc_baselines/data")

TRANCHES = ["BBAB", "FBAB", "JBCD"]
PROPERTY_COLUMNS = ["QED", "HeavyAtomMolWt", "MolWt", "TPSA", "MolLogP", "num_atoms"]


def load_and_prepare_tranche(tranche_name: str, data_dir: Path) -> dict:
    """Load a ZINC tranche and prepare for torch-molecule."""
    
    # Load the CSV file - adjust filename pattern as needed
    csv_path = data_dir / f"{tranche_name}.csv"
    if not csv_path.exists():
        # Try alternative naming conventions
        for pattern in [f"zinc_{tranche_name}.csv", f"{tranche_name.lower()}.csv"]:
            alt_path = data_dir / pattern
            if alt_path.exists():
                csv_path = alt_path
                break
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find data file for tranche {tranche_name} in {data_dir}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {tranche_name}: {len(df)} molecules")
    
    # Ensure required columns exist
    assert "smiles" in df.columns, f"Missing 'smiles' column in {tranche_name}"
    assert "split" in df.columns, f"Missing 'split' column in {tranche_name}"
    for col in PROPERTY_COLUMNS:
        assert col in df.columns, f"Missing '{col}' column in {tranche_name}"
    
    # Split data
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Prepare outputs
    data = {
        "tranche": tranche_name,
        "X_train": train_df["smiles"].tolist(),
        "X_val": val_df["smiles"].tolist(),
        "X_test": test_df["smiles"].tolist(),
        "y_train": train_df[PROPERTY_COLUMNS].values.astype(np.float32),
        "y_val": val_df[PROPERTY_COLUMNS].values.astype(np.float32),
        "y_test": test_df[PROPERTY_COLUMNS].values.astype(np.float32),
        "property_names": PROPERTY_COLUMNS,
        # Statistics for normalization (computed on training set)
        "property_means": train_df[PROPERTY_COLUMNS].mean().values.astype(np.float32),
        "property_stds": train_df[PROPERTY_COLUMNS].std().values.astype(np.float32),
    }
    
    return data


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for tranche in TRANCHES:
        print(f"\nProcessing {tranche}...")
        try:
            data = load_and_prepare_tranche(tranche, DATA_DIR)
            
            # Save as pickle for easy loading
            output_path = OUTPUT_DIR / f"{tranche}_prepared.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(data, f)
            print(f"  Saved to {output_path}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
```

### 3.3 Run Data Preparation
```bash
# Update DATA_DIR in the script first!
python experiments/zinc_baselines/scripts/prepare_data.py
```

---

## Step 4: Training Scripts

### 4.1 Unconditional Generation Training

Create `experiments/zinc_baselines/scripts/train_unconditional.py`:

```python
"""
Train Graph DiT and DiGress for unconditional molecular generation.
Usage: python train_unconditional.py --model graph_dit --tranche BBAB
"""
import argparse
import pickle
from pathlib import Path
import torch
import numpy as np

from torch_molecule import GraphDiTMolecularGenerator, DigressMolecularGenerator


def load_data(tranche: str, data_dir: Path) -> dict:
    """Load prepared data for a tranche."""
    data_path = data_dir / f"{tranche}_prepared.pkl"
    with open(data_path, "rb") as f:
        return pickle.load(f)


def train_graph_dit_unconditional(data: dict, output_dir: Path, **kwargs):
    """Train Graph DiT for unconditional generation."""
    
    model = GraphDiTMolecularGenerator(
        # Model architecture
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        # Diffusion settings
        diffusion_steps=500,
        noise_schedule="cosine",
        # Training settings
        batch_size=kwargs.get("batch_size", 64),
        learning_rate=kwargs.get("lr", 1e-4),
        max_epochs=kwargs.get("epochs", 1000),
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Logging
        verbose="progress_bar",
    )
    
    # For unconditional generation, pass y=None
    model.fit(
        X_train=data["X_train"],
        y_train=None,  # Unconditional
        X_val=data["X_val"],
        y_val=None,
    )
    
    # Save model
    checkpoint_path = output_dir / f"graph_dit_{data['tranche']}_unconditional.pt"
    model.save_to_local(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")
    
    return model


def train_digress_unconditional(data: dict, output_dir: Path, **kwargs):
    """Train DiGress for unconditional generation."""
    
    model = DigressMolecularGenerator(
        # Model architecture
        hidden_dim=256,
        num_layers=6,
        # Diffusion settings  
        diffusion_steps=500,
        noise_schedule="cosine",
        # Training settings
        batch_size=kwargs.get("batch_size", 64),
        learning_rate=kwargs.get("lr", 1e-4),
        max_epochs=kwargs.get("epochs", 1000),
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Logging
        verbose="progress_bar",
    )
    
    model.fit(
        X_train=data["X_train"],
        y_train=None,
        X_val=data["X_val"],
        y_val=None,
    )
    
    checkpoint_path = output_dir / f"digress_{data['tranche']}_unconditional.pt"
    model.save_to_local(str(checkpoint_path))
    print(f"Model saved to {checkpoint_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["graph_dit", "digress"], required=True)
    parser.add_argument("--tranche", choices=["BBAB", "FBAB", "JBCD"], required=True)
    parser.add_argument("--data_dir", type=str, default="experiments/zinc_baselines/data")
    parser.add_argument("--output_dir", type=str, default="experiments/zinc_baselines/checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data for {args.tranche}...")
    data = load_data(args.tranche, data_dir)
    
    print(f"Training {args.model} on {args.tranche} (unconditional)...")
    if args.model == "graph_dit":
        train_graph_dit_unconditional(data, output_dir, 
                                       batch_size=args.batch_size,
                                       lr=args.lr, 
                                       epochs=args.epochs)
    else:
        train_digress_unconditional(data, output_dir,
                                     batch_size=args.batch_size,
                                     lr=args.lr,
                                     epochs=args.epochs)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
```

### 4.2 Conditional Generation Training (Multi-Property)

Create `experiments/zinc_baselines/scripts/train_conditional.py`:

```python
"""
Train Graph DiT and DiGress for multi-property conditional generation.
Properties: QED, HeavyAtomMolWt, MolWt, TPSA, MolLogP, num_atoms

Usage: python train_conditional.py --model graph_dit --tranche BBAB
"""
import argparse
import pickle
from pathlib import Path
import torch
import numpy as np

from torch_molecule import GraphDiTMolecularGenerator, DigressMolecularGenerator


def load_data(tranche: str, data_dir: Path) -> dict:
    """Load prepared data for a tranche."""
    data_path = data_dir / f"{tranche}_prepared.pkl"
    with open(data_path, "rb") as f:
        return pickle.load(f)


def train_graph_dit_conditional(data: dict, output_dir: Path, **kwargs):
    """
    Train Graph DiT for multi-property conditional generation.
    
    Graph DiT uses AdaLN (Adaptive Layer Normalization) for conditioning,
    which is their recommended approach for numerical properties.
    """
    
    num_properties = len(data["property_names"])
    
    model = GraphDiTMolecularGenerator(
        # Model architecture
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        # Conditioning - Graph DiT specific
        num_classes=num_properties,  # Number of conditioning properties
        class_dropout_prob=0.1,  # For classifier-free guidance
        # Diffusion settings
        diffusion_steps=500,
        noise_schedule="cosine",
        # Training settings
        batch_size=kwargs.get("batch_size", 64),
        learning_rate=kwargs.get("lr", 1e-4),
        max_epochs=kwargs.get("epochs", 1000),
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Logging
        verbose="progress_bar",
    )
    
    # Normalize properties using training set statistics
    y_train_normalized = (data["y_train"] - data["property_means"]) / (data["property_stds"] + 1e-8)
    y_val_normalized = (data["y_val"] - data["property_means"]) / (data["property_stds"] + 1e-8)
    
    model.fit(
        X_train=data["X_train"],
        y_train=y_train_normalized,
        X_val=data["X_val"],
        y_val=y_val_normalized,
    )
    
    # Save model
    checkpoint_path = output_dir / f"graph_dit_{data['tranche']}_conditional.pt"
    model.save_to_local(str(checkpoint_path))
    
    # Save normalization statistics for inference
    stats_path = output_dir / f"graph_dit_{data['tranche']}_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump({
            "property_names": data["property_names"],
            "property_means": data["property_means"],
            "property_stds": data["property_stds"],
        }, f)
    
    print(f"Model saved to {checkpoint_path}")
    print(f"Stats saved to {stats_path}")
    
    return model


def train_digress_conditional(data: dict, output_dir: Path, **kwargs):
    """
    Train DiGress for multi-property conditional generation.
    
    DiGress typically uses a property predictor for guidance during generation.
    """
    
    num_properties = len(data["property_names"])
    
    model = DigressMolecularGenerator(
        # Model architecture
        hidden_dim=256,
        num_layers=6,
        # Conditioning
        num_classes=num_properties,
        # Diffusion settings
        diffusion_steps=500,
        noise_schedule="cosine",
        # Training settings
        batch_size=kwargs.get("batch_size", 64),
        learning_rate=kwargs.get("lr", 1e-4),
        max_epochs=kwargs.get("epochs", 1000),
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Logging
        verbose="progress_bar",
    )
    
    # Normalize properties
    y_train_normalized = (data["y_train"] - data["property_means"]) / (data["property_stds"] + 1e-8)
    y_val_normalized = (data["y_val"] - data["property_means"]) / (data["property_stds"] + 1e-8)
    
    model.fit(
        X_train=data["X_train"],
        y_train=y_train_normalized,
        X_val=data["X_val"],
        y_val=y_val_normalized,
    )
    
    checkpoint_path = output_dir / f"digress_{data['tranche']}_conditional.pt"
    model.save_to_local(str(checkpoint_path))
    
    stats_path = output_dir / f"digress_{data['tranche']}_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump({
            "property_names": data["property_names"],
            "property_means": data["property_means"],
            "property_stds": data["property_stds"],
        }, f)
    
    print(f"Model saved to {checkpoint_path}")
    print(f"Stats saved to {stats_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["graph_dit", "digress"], required=True)
    parser.add_argument("--tranche", choices=["BBAB", "FBAB", "JBCD"], required=True)
    parser.add_argument("--data_dir", type=str, default="experiments/zinc_baselines/data")
    parser.add_argument("--output_dir", type=str, default="experiments/zinc_baselines/checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data for {args.tranche}...")
    data = load_data(args.tranche, data_dir)
    print(f"Properties: {data['property_names']}")
    
    print(f"Training {args.model} on {args.tranche} (conditional on {len(data['property_names'])} properties)...")
    if args.model == "graph_dit":
        train_graph_dit_conditional(data, output_dir,
                                     batch_size=args.batch_size,
                                     lr=args.lr,
                                     epochs=args.epochs)
    else:
        train_digress_conditional(data, output_dir,
                                   batch_size=args.batch_size,
                                   lr=args.lr,
                                   epochs=args.epochs)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
```

---

## Step 5: Evaluation Scripts

### 5.1 Evaluation Utilities

Create `experiments/zinc_baselines/scripts/evaluate.py`:

```python
"""
Evaluation script for generated molecules.
Computes metrics for Table 1 (unconditional) and Table 2 (conditional).

Usage:
  python evaluate.py --model graph_dit --tranche BBAB --mode unconditional
  python evaluate.py --model graph_dit --tranche BBAB --mode conditional
"""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import DataStructs
from collections import Counter
import torch

# Import evaluation utilities
try:
    from fcd_torch import FCD
    FCD_AVAILABLE = True
except ImportError:
    FCD_AVAILABLE = False
    print("Warning: fcd_torch not available, FCD metric will be skipped")

try:
    from mini_moses import metrics as moses_metrics
    MOSES_AVAILABLE = True
except ImportError:
    MOSES_AVAILABLE = False
    print("Warning: mini_moses not available, some metrics may be limited")

from torch_molecule import GraphDiTMolecularGenerator, DigressMolecularGenerator


def load_model(model_type: str, checkpoint_path: Path):
    """Load a trained model from checkpoint."""
    if model_type == "graph_dit":
        model = GraphDiTMolecularGenerator()
    else:
        model = DigressMolecularGenerator()
    
    model.load_from_local(str(checkpoint_path))
    return model


def compute_validity(smiles_list: list) -> tuple:
    """Compute validity rate and return valid SMILES."""
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Canonicalize
            valid_smiles.append(Chem.MolToSmiles(mol))
    
    validity = len(valid_smiles) / len(smiles_list) if smiles_list else 0
    return validity, valid_smiles


def compute_uniqueness(smiles_list: list) -> float:
    """Compute uniqueness among valid SMILES."""
    if not smiles_list:
        return 0.0
    unique = set(smiles_list)
    return len(unique) / len(smiles_list)


def compute_novelty(generated_smiles: list, train_smiles: list) -> float:
    """Compute novelty (fraction not in training set)."""
    if not generated_smiles:
        return 0.0
    train_set = set(train_smiles)
    novel = [s for s in generated_smiles if s not in train_set]
    return len(novel) / len(generated_smiles)


def compute_diversity(smiles_list: list, sample_size: int = 1000) -> float:
    """Compute internal diversity using Tanimoto similarity."""
    if len(smiles_list) < 2:
        return 0.0
    
    # Sample if too many
    if len(smiles_list) > sample_size:
        smiles_list = np.random.choice(smiles_list, sample_size, replace=False).tolist()
    
    # Compute fingerprints
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    
    if len(fps) < 2:
        return 0.0
    
    # Compute pairwise similarities
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarities.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    
    # Diversity = 1 - average similarity
    return 1.0 - np.mean(similarities)


def compute_similarity_to_training(generated_smiles: list, train_smiles: list, sample_size: int = 1000) -> float:
    """Compute average nearest-neighbor similarity to training set."""
    if not generated_smiles or not train_smiles:
        return 0.0
    
    # Sample if needed
    if len(generated_smiles) > sample_size:
        generated_smiles = np.random.choice(generated_smiles, sample_size, replace=False).tolist()
    if len(train_smiles) > sample_size:
        train_smiles = np.random.choice(train_smiles, sample_size, replace=False).tolist()
    
    # Compute fingerprints
    gen_fps = []
    for smi in generated_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            gen_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    
    train_fps = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    
    if not gen_fps or not train_fps:
        return 0.0
    
    # For each generated molecule, find max similarity to training set
    nn_sims = []
    for gfp in gen_fps:
        sims = [DataStructs.TanimotoSimilarity(gfp, tfp) for tfp in train_fps]
        nn_sims.append(max(sims))
    
    return np.mean(nn_sims)


def compute_fcd(generated_smiles: list, reference_smiles: list) -> float:
    """Compute Fréchet ChemNet Distance."""
    if not FCD_AVAILABLE:
        return -1.0
    
    fcd = FCD(device="cuda" if torch.cuda.is_available() else "cpu")
    return fcd(generated_smiles, reference_smiles)


def compute_property_metrics(generated_smiles: list, target_properties: np.ndarray, 
                             property_names: list, property_means: np.ndarray,
                             property_stds: np.ndarray) -> dict:
    """
    Compute property-related metrics for conditional generation.
    Returns MAE and z-scores for each property.
    """
    results = {}
    
    # Compute actual properties for generated molecules
    computed_props = []
    valid_indices = []
    
    for i, smi in enumerate(generated_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        props = []
        for pname in property_names:
            if pname == "QED":
                props.append(Descriptors.qed(mol))
            elif pname == "HeavyAtomMolWt":
                props.append(Descriptors.HeavyAtomMolWt(mol))
            elif pname == "MolWt":
                props.append(Descriptors.MolWt(mol))
            elif pname == "TPSA":
                props.append(Descriptors.TPSA(mol))
            elif pname == "MolLogP":
                props.append(Descriptors.MolLogP(mol))
            elif pname == "num_atoms":
                props.append(mol.GetNumAtoms())
            else:
                props.append(0.0)  # Unknown property
        
        computed_props.append(props)
        valid_indices.append(i)
    
    if not computed_props:
        return {"error": "No valid molecules"}
    
    computed_props = np.array(computed_props)
    target_subset = target_properties[valid_indices]
    
    # Denormalize targets if they were normalized
    target_denorm = target_subset * property_stds + property_means
    
    # Compute metrics per property
    for i, pname in enumerate(property_names):
        computed = computed_props[:, i]
        target = target_denorm[:, i]
        
        mae = np.mean(np.abs(computed - target))
        
        # Z-score: how many std deviations the generated mean is from training mean
        z_score = (np.mean(computed) - property_means[i]) / (property_stds[i] + 1e-8)
        
        results[f"{pname}_mae"] = mae
        results[f"{pname}_mean"] = np.mean(computed)
        results[f"{pname}_std"] = np.std(computed)
        results[f"{pname}_zscore"] = z_score
    
    return results


def evaluate_unconditional(model, data: dict, n_samples: int = 10000) -> dict:
    """Evaluate unconditional generation (Table 1 metrics)."""
    print(f"Generating {n_samples} molecules...")
    generated_smiles = model.generate(num_samples=n_samples)
    
    print("Computing metrics...")
    validity, valid_smiles = compute_validity(generated_smiles)
    uniqueness = compute_uniqueness(valid_smiles)
    novelty = compute_novelty(valid_smiles, data["X_train"])
    diversity = compute_diversity(valid_smiles)
    similarity = compute_similarity_to_training(valid_smiles, data["X_train"])
    fcd = compute_fcd(valid_smiles, data["X_test"])
    
    results = {
        "n_generated": len(generated_smiles),
        "n_valid": len(valid_smiles),
        "validity": validity * 100,
        "uniqueness": uniqueness * 100,
        "novelty": novelty * 100,
        "diversity": diversity * 100,
        "similarity": similarity * 100,
        "fcd": fcd,
    }
    
    return results


def evaluate_conditional(model, data: dict, stats: dict, n_samples: int = 10000) -> dict:
    """Evaluate conditional generation (Table 2 metrics)."""
    
    # Sample target conditions from test set
    n_test = len(data["X_test"])
    if n_samples > n_test:
        # Repeat test conditions
        indices = np.random.choice(n_test, n_samples, replace=True)
    else:
        indices = np.random.choice(n_test, n_samples, replace=False)
    
    # Normalize target properties
    target_props = data["y_test"][indices]
    target_normalized = (target_props - stats["property_means"]) / (stats["property_stds"] + 1e-8)
    
    print(f"Generating {n_samples} molecules with target conditions...")
    generated_smiles = model.generate(
        num_samples=n_samples,
        conditions=target_normalized,
    )
    
    print("Computing metrics...")
    validity, valid_smiles = compute_validity(generated_smiles)
    
    # Property metrics
    prop_metrics = compute_property_metrics(
        generated_smiles,
        target_normalized,
        stats["property_names"],
        stats["property_means"],
        stats["property_stds"],
    )
    
    results = {
        "n_generated": len(generated_smiles),
        "n_valid": len(valid_smiles),
        "validity": validity * 100,
        **prop_metrics,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["graph_dit", "digress"], required=True)
    parser.add_argument("--tranche", choices=["BBAB", "FBAB", "JBCD"], required=True)
    parser.add_argument("--mode", choices=["unconditional", "conditional"], required=True)
    parser.add_argument("--data_dir", type=str, default="experiments/zinc_baselines/data")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/zinc_baselines/checkpoints")
    parser.add_argument("--output_dir", type=str, default="experiments/zinc_baselines/results")
    parser.add_argument("--n_samples", type=int, default=10000)
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data for {args.tranche}...")
    data_path = data_dir / f"{args.tranche}_prepared.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # Load model
    suffix = "unconditional" if args.mode == "unconditional" else "conditional"
    checkpoint_path = checkpoint_dir / f"{args.model}_{args.tranche}_{suffix}.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(args.model, checkpoint_path)
    
    # Evaluate
    if args.mode == "unconditional":
        results = evaluate_unconditional(model, data, args.n_samples)
    else:
        # Load stats for denormalization
        stats_path = checkpoint_dir / f"{args.model}_{args.tranche}_stats.pkl"
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        results = evaluate_conditional(model, data, stats, args.n_samples)
    
    # Save results
    results_path = output_dir / f"{args.tranche}_{args.mode}_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    
    # Print results
    print("\n" + "="*60)
    print(f"RESULTS: {args.model} on {args.tranche} ({args.mode})")
    print("="*60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Save as CSV for easy LaTeX conversion
    csv_path = output_dir / f"{args.tranche}_{args.mode}_results.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    print(f"\nResults saved to {results_path} and {csv_path}")


if __name__ == "__main__":
    main()
```

---

## Step 6: SLURM Job Scripts (for HUJI Moriah Cluster)

Create `experiments/zinc_baselines/scripts/submit_training.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=baseline_train
#SBATCH --partition=killable  # or your preferred partition (e.g., gpu-a100)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch submit_training.sh graph_dit BBAB unconditional

MODEL=$1
TRANCHE=$2
MODE=$3

# Activate environment
source ~/.bashrc
conda activate torch_molecule

# Create logs directory
mkdir -p logs

# Run training
if [ "$MODE" == "unconditional" ]; then
    python experiments/zinc_baselines/scripts/train_unconditional.py \
        --model $MODEL \
        --tranche $TRANCHE \
        --epochs 1000 \
        --batch_size 64
else
    python experiments/zinc_baselines/scripts/train_conditional.py \
        --model $MODEL \
        --tranche $TRANCHE \
        --epochs 1000 \
        --batch_size 64
fi
```

Create `experiments/zinc_baselines/scripts/submit_all_training.sh`:

```bash
#!/bin/bash
# Submit all training jobs

mkdir -p logs

for MODEL in graph_dit digress; do
    for TRANCHE in BBAB FBAB JBCD; do
        for MODE in unconditional conditional; do
            echo "Submitting: $MODEL $TRANCHE $MODE"
            sbatch submit_training.sh $MODEL $TRANCHE $MODE
            sleep 1  # Avoid overwhelming scheduler
        done
    done
done

echo "All jobs submitted!"
```

---

## Step 7: Running Everything

### 7.1 Full Workflow

```bash
# 1. Prepare data (update paths in script first!)
python experiments/zinc_baselines/scripts/prepare_data.py

# 2. Submit all training jobs
cd experiments/zinc_baselines/scripts
chmod +x submit_all_training.sh
./submit_all_training.sh

# Or run individually:
sbatch submit_training.sh graph_dit BBAB unconditional
sbatch submit_training.sh graph_dit BBAB conditional
# ... etc

# 3. After training completes, run evaluation
for MODEL in graph_dit digress; do
    for TRANCHE in BBAB FBAB JBCD; do
        python experiments/zinc_baselines/scripts/evaluate.py \
            --model $MODEL --tranche $TRANCHE --mode unconditional
        python experiments/zinc_baselines/scripts/evaluate.py \
            --model $MODEL --tranche $TRANCHE --mode conditional
    done
done

# 4. Aggregate results
python experiments/zinc_baselines/scripts/aggregate_results.py
```

### 7.2 Expected Output Structure

After running everything:

```
experiments/zinc_baselines/
├── data/
│   ├── BBAB_prepared.pkl
│   ├── FBAB_prepared.pkl
│   └── JBCD_prepared.pkl
├── checkpoints/
│   ├── graph_dit_BBAB_unconditional.pt
│   ├── graph_dit_BBAB_conditional.pt
│   ├── graph_dit_BBAB_stats.pkl
│   ├── digress_BBAB_unconditional.pt
│   ├── digress_BBAB_conditional.pt
│   ├── digress_BBAB_stats.pkl
│   └── ... (same for FBAB, JBCD)
├── results/
│   ├── graph_dit/
│   │   ├── BBAB_unconditional_results.csv
│   │   ├── BBAB_conditional_results.csv
│   │   └── ...
│   └── digress/
│       ├── BBAB_unconditional_results.csv
│       ├── BBAB_conditional_results.csv
│       └── ...
└── scripts/
    └── ...
```

---

## Step 8: Results Aggregation for LaTeX

Create `experiments/zinc_baselines/scripts/aggregate_results.py`:

```python
"""
Aggregate results into LaTeX-ready tables.
"""
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("experiments/zinc_baselines/results")
OUTPUT_DIR = Path("experiments/zinc_baselines/results")

MODELS = ["graph_dit", "digress"]
TRANCHES = ["BBAB", "FBAB", "JBCD"]


def aggregate_unconditional():
    """Create Table 1: Unconditional generation metrics."""
    rows = []
    
    for tranche in TRANCHES:
        for model in MODELS:
            csv_path = RESULTS_DIR / model / f"{tranche}_unconditional_results.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                row = df.iloc[0].to_dict()
                row["model"] = model
                row["tranche"] = tranche
                rows.append(row)
    
    if not rows:
        print("No unconditional results found!")
        return
    
    results_df = pd.DataFrame(rows)
    
    # Format for LaTeX
    latex_cols = ["tranche", "model", "validity", "uniqueness", "novelty", "diversity", "similarity", "fcd"]
    latex_df = results_df[latex_cols].copy()
    
    print("\n=== TABLE 1: Unconditional Generation ===")
    print(latex_df.to_latex(index=False, float_format="%.2f"))
    
    latex_df.to_csv(OUTPUT_DIR / "table1_unconditional.csv", index=False)


def aggregate_conditional():
    """Create Table 2: Conditional generation metrics."""
    rows = []
    
    for tranche in TRANCHES:
        for model in MODELS:
            csv_path = RESULTS_DIR / model / f"{tranche}_conditional_results.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                row = df.iloc[0].to_dict()
                row["model"] = model
                row["tranche"] = tranche
                rows.append(row)
    
    if not rows:
        print("No conditional results found!")
        return
    
    results_df = pd.DataFrame(rows)
    
    # Select key metrics for the table
    key_cols = ["tranche", "model", "validity"]
    # Add z-scores for each property
    for prop in ["QED", "MolLogP"]:
        if f"{prop}_zscore" in results_df.columns:
            key_cols.append(f"{prop}_zscore")
    
    latex_df = results_df[key_cols].copy()
    
    print("\n=== TABLE 2: Conditional Generation ===")
    print(latex_df.to_latex(index=False, float_format="%.2f"))
    
    latex_df.to_csv(OUTPUT_DIR / "table2_conditional.csv", index=False)


def main():
    aggregate_unconditional()
    aggregate_conditional()
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` (try 32 or 16)
   - Use gradient checkpointing if available

2. **Import Errors**
   - Ensure torch-molecule is installed in editable mode: `pip install -e .`
   - Check PyTorch Geometric compatibility

3. **Slow Training**
   - Enable mixed precision if supported
   - Use DataLoader with `num_workers > 0`

4. **Invalid Molecules**
   - This is expected for diffusion models
   - Report validity rate in your results

### Verifying torch-molecule API

If the sklearn-style API differs from what's documented here, check:
```python
from torch_molecule import GraphDiTMolecularGenerator
help(GraphDiTMolecularGenerator)
help(GraphDiTMolecularGenerator.fit)
help(GraphDiTMolecularGenerator.generate)
```

---

## Notes for the Agent

1. **API Discovery**: The torch-molecule package uses sklearn-style `.fit()` and `.generate()` methods. If parameters differ from this document, inspect the actual class signatures.

2. **Conditioning Mechanism**: 
   - Graph DiT uses AdaLN (Adaptive Layer Normalization) for property conditioning
   - DiGress may use classifier guidance or similar mechanisms
   - Both support multi-property conditioning via numpy arrays

3. **Evaluation Consistency**: Use the same evaluation code for both baselines AND SCULPT to ensure fair comparison.

4. **Property Normalization**: Always normalize properties using training set statistics for conditional generation.

5. **Checkpoints**: Save both model weights and normalization statistics for reproducibility.

---

## Contact

If you encounter issues with torch-molecule internals, refer to:
- torch-molecule docs: https://liugangcode.github.io/torch-molecule/
- Original Graph DiT repo: https://github.com/liugangcode/Graph-DiT
- Original DiGress paper implementation references in torch-molecule


# IMPORTANT
Work incrementally. After completing each major step, pause and report your progress before continuing. When you encounter multiple valid approaches, API mismatches, or unclear requirements, ask me before proceeding.