"""
Evaluation script for generated molecules.
Computes metrics for Table 1 (unconditional) and Table 2 (conditional).

Table 1 Metrics: Validity, Uniqueness, Novelty, Diversity, Similarity, FCD
Table 2 Metrics: Property MAE, validity under conditioning

Usage:
    python experiments/zinc_baselines/scripts/evaluate.py --model graph_dit --tranche BBAB --mode unconditional
    python experiments/zinc_baselines/scripts/evaluate.py --model graph_dit --tranche BBAB --mode conditional
    python experiments/zinc_baselines/scripts/evaluate.py --model digress --tranche BBAB --mode unconditional
"""
import argparse
import pickle
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED, rdMolDescriptors
from rdkit import DataStructs

# Import generators
from torch_molecule.generator.graph_dit import GraphDITMolecularGenerator
from torch_molecule.generator.digress import DigressMolecularGenerator

# Try to import FCD
try:
    from fcd_torch import FCD
    FCD_AVAILABLE = True
except ImportError:
    FCD_AVAILABLE = False
    print("Warning: fcd_torch not available, FCD metric will be skipped")


def load_model(model_type: str, checkpoint_path: Path):
    """Load a trained model from checkpoint."""
    if model_type == "graph_dit":
        model = GraphDITMolecularGenerator(device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        model = DigressMolecularGenerator(device="cuda" if torch.cuda.is_available() else "cpu")
    
    model.load_from_local(str(checkpoint_path))
    return model


def compute_validity(smiles_list: list) -> tuple:
    """Compute validity rate and return valid SMILES."""
    valid_smiles = []
    for smi in smiles_list:
        if smi is None:
            continue
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
    
    try:
        fcd = FCD(device="cuda" if torch.cuda.is_available() else "cpu")
        return fcd(generated_smiles, reference_smiles)
    except Exception as e:
        print(f"FCD computation failed: {e}")
        return -1.0


def compute_property_from_smiles(smi: str, property_names: list) -> dict:
    """Compute molecular properties for a single SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {p: float('nan') for p in property_names}
    
    props = {}
    for pname in property_names:
        try:
            if pname == "qed":
                props[pname] = QED.qed(mol)
            elif pname == "HeavyAtomMolWt":
                props[pname] = Descriptors.HeavyAtomMolWt(mol)
            elif pname == "MolWt":
                props[pname] = Descriptors.MolWt(mol)
            elif pname == "TPSA":
                props[pname] = rdMolDescriptors.CalcTPSA(mol)
            elif pname == "MolLogP":
                props[pname] = Descriptors.MolLogP(mol)
            elif pname == "num_atoms":
                props[pname] = float(mol.GetNumHeavyAtoms())
            else:
                props[pname] = float('nan')
        except:
            props[pname] = float('nan')
    
    return props


def compute_property_metrics(generated_smiles: list, target_properties: np.ndarray,
                             property_names: list, property_means: np.ndarray,
                             property_stds: np.ndarray) -> dict:
    """
    Compute property-related metrics for conditional generation.
    Returns MAE for each property.
    """
    results = {}
    
    # Compute actual properties for generated molecules
    computed_props = []
    valid_indices = []
    
    for i, smi in enumerate(tqdm(generated_smiles, desc="Computing properties")):
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        props = compute_property_from_smiles(smi, property_names)
        prop_values = [props[p] for p in property_names]
        
        if not any(np.isnan(prop_values)):
            computed_props.append(prop_values)
            valid_indices.append(i)
    
    if not computed_props:
        return {"error": "No valid molecules"}
    
    computed_props = np.array(computed_props)
    target_subset = target_properties[valid_indices]
    
    # Denormalize targets (they were normalized during training)
    target_denorm = target_subset * property_stds + property_means
    
    # Compute metrics per property
    for i, pname in enumerate(property_names):
        computed = computed_props[:, i]
        target = target_denorm[:, i]
        
        mae = np.mean(np.abs(computed - target))
        rmse = np.sqrt(np.mean((computed - target) ** 2))
        
        results[f"{pname}_mae"] = mae
        results[f"{pname}_rmse"] = rmse
        results[f"{pname}_mean_generated"] = np.mean(computed)
        results[f"{pname}_mean_target"] = np.mean(target)
    
    results["n_valid_for_props"] = len(computed_props)
    
    return results


def evaluate_unconditional(model, data: dict, n_samples: int = 10000, batch_size: int = 100) -> dict:
    """Evaluate unconditional generation (Table 1 metrics)."""
    
    print(f"\nGenerating {n_samples} molecules...")
    
    # Generate in batches
    all_smiles = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Generating"):
        current_batch_size = min(batch_size, n_samples - len(all_smiles))
        try:
            batch_smiles = model.generate(batch_size=current_batch_size)
            all_smiles.extend(batch_smiles)
        except Exception as e:
            print(f"Generation error in batch {i}: {e}")
            continue
    
    print(f"Generated {len(all_smiles)} molecules")
    
    print("\nComputing metrics...")
    validity, valid_smiles = compute_validity(all_smiles)
    print(f"  Validity: {validity*100:.2f}% ({len(valid_smiles)} valid)")
    
    uniqueness = compute_uniqueness(valid_smiles)
    print(f"  Uniqueness: {uniqueness*100:.2f}%")
    
    novelty = compute_novelty(valid_smiles, data["X_train"])
    print(f"  Novelty: {novelty*100:.2f}%")
    
    diversity = compute_diversity(valid_smiles)
    print(f"  Diversity: {diversity*100:.2f}%")
    
    similarity = compute_similarity_to_training(valid_smiles, data["X_train"])
    print(f"  Similarity to train: {similarity*100:.2f}%")
    
    fcd = compute_fcd(valid_smiles, data["X_test"])
    print(f"  FCD: {fcd:.4f}")
    
    results = {
        "n_generated": len(all_smiles),
        "n_valid": len(valid_smiles),
        "validity": validity * 100,
        "uniqueness": uniqueness * 100,
        "novelty": novelty * 100,
        "diversity": diversity * 100,
        "similarity": similarity * 100,
        "fcd": fcd,
    }
    
    return results


def evaluate_conditional(model, data: dict, stats: dict, n_samples: int = 10000, batch_size: int = 100) -> dict:
    """Evaluate conditional generation (Table 2 metrics)."""
    
    # Sample target conditions from test set
    n_test = len(data["X_test"])
    if n_samples > n_test:
        indices = np.random.choice(n_test, n_samples, replace=True)
    else:
        indices = np.random.choice(n_test, n_samples, replace=False)
    
    # Get target properties and normalize
    target_props = data["y_test"][indices]
    target_normalized = (target_props - stats["property_means"]) / (stats["property_stds"] + 1e-8)
    target_normalized = np.nan_to_num(target_normalized, nan=0.0)
    
    print(f"\nGenerating {n_samples} molecules with target conditions...")
    
    # Generate in batches
    all_smiles = []
    all_targets = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(n_batches), desc="Generating"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch_targets = target_normalized[start_idx:end_idx]
        
        try:
            batch_smiles = model.generate(labels=batch_targets, batch_size=len(batch_targets))
            all_smiles.extend(batch_smiles)
            all_targets.append(batch_targets)
        except Exception as e:
            print(f"Generation error in batch {i}: {e}")
            continue
    
    all_targets = np.vstack(all_targets) if all_targets else np.array([])
    
    print(f"Generated {len(all_smiles)} molecules")
    
    print("\nComputing metrics...")
    validity, valid_smiles = compute_validity(all_smiles)
    print(f"  Validity: {validity*100:.2f}% ({len(valid_smiles)} valid)")
    
    # Property metrics
    prop_metrics = compute_property_metrics(
        all_smiles,
        all_targets,
        stats["property_names"],
        stats["property_means"],
        stats["property_stds"],
    )
    
    for key, value in prop_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    results = {
        "n_generated": len(all_smiles),
        "n_valid": len(valid_smiles),
        "validity": validity * 100,
        **prop_metrics,
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecules")
    parser.add_argument("--model", choices=["graph_dit", "digress"], required=True)
    parser.add_argument("--tranche", choices=["BBAB", "FBAB", "JBCD"], required=True)
    parser.add_argument("--mode", choices=["unconditional", "conditional"], required=True)
    parser.add_argument("--data_dir", type=str, default="experiments/zinc_baselines/data")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/zinc_baselines/checkpoints")
    parser.add_argument("--output_dir", type=str, default="experiments/zinc_baselines/results")
    parser.add_argument("--n_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=100)
    
    args = parser.parse_args()
    
    # Validate: DiGress only supports unconditional
    if args.model == "digress" and args.mode == "conditional":
        print("ERROR: DiGress only supports unconditional generation.")
        print("Use GraphDIT for conditional generation.")
        return
    
    # Resolve paths
    if not Path(args.data_dir).is_absolute():
        data_dir = PROJECT_ROOT / args.data_dir
    else:
        data_dir = Path(args.data_dir)
        
    if not Path(args.checkpoint_dir).is_absolute():
        checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        
    if not Path(args.output_dir).is_absolute():
        output_dir = PROJECT_ROOT / args.output_dir / args.model
    else:
        output_dir = Path(args.output_dir) / args.model
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"Evaluation: {args.model} on {args.tranche} ({args.mode})")
    print(f"{'='*60}")
    
    # Load data
    print(f"\nLoading data for {args.tranche}...")
    data_path = data_dir / f"{args.tranche}_prepared.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # Load model
    checkpoint_path = checkpoint_dir / f"{args.model}_{args.tranche}_{args.mode}.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(args.model, checkpoint_path)
    
    # Evaluate
    if args.mode == "unconditional":
        results = evaluate_unconditional(model, data, args.n_samples, args.batch_size)
    else:
        # Load stats for denormalization
        stats_path = checkpoint_dir / f"{args.model}_{args.tranche}_conditional_stats.pkl"
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        results = evaluate_conditional(model, data, stats, args.n_samples, args.batch_size)
    
    # Save results
    results_path = output_dir / f"{args.tranche}_{args.mode}_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model} on {args.tranche} ({args.mode})")
    print(f"{'='*60}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Save as CSV
    csv_path = output_dir / f"{args.tranche}_{args.mode}_results.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    print(f"\nResults saved to {results_path} and {csv_path}")


if __name__ == "__main__":
    main()
