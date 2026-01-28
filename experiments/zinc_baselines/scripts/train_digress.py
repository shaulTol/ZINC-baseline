"""
Train DiGress for unconditional molecular generation.

Note: DiGress in torch_molecule only supports unconditional generation.
For conditional generation, use GraphDIT instead.

Usage:
    python experiments/zinc_baselines/scripts/train_digress.py --tranche BBAB
"""
import argparse
import pickle
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np

from torch_molecule.generator.digress import DigressMolecularGenerator


def load_data(tranche: str, data_dir: Path) -> dict:
    """Load prepared data for a tranche."""
    data_path = data_dir / f"{tranche}_prepared.pkl"
    with open(data_path, "rb") as f:
        return pickle.load(f)


def train_unconditional(data: dict, output_dir: Path, **kwargs):
    """Train DiGress for unconditional generation."""
    
    print(f"\nInitializing DiGress for unconditional generation...")
    print(f"  hidden_size_X: {kwargs.get('hidden_size_X', 256)}")
    print(f"  hidden_size_E: {kwargs.get('hidden_size_E', 128)}")
    print(f"  num_layer: {kwargs.get('num_layer', 12)}")
    print(f"  epochs: {kwargs.get('epochs', 2000)}")
    print(f"  batch_size: {kwargs.get('batch_size', 16)}")
    
    model = DigressMolecularGenerator(
        # Model architecture (matched to GRASSY-DiT for fair comparison)
        hidden_size_X=kwargs.get("hidden_size_X", 256),
        hidden_size_E=kwargs.get("hidden_size_E", 128),
        num_layer=kwargs.get("num_layer", 12),
        n_head=kwargs.get("n_head", 16),
        dropout=kwargs.get("dropout", 0.1),
        # Diffusion settings
        timesteps=kwargs.get("timesteps", 500),
        # Training settings
        batch_size=kwargs.get("batch_size", 16),
        learning_rate=kwargs.get("lr", 1e-4),
        epochs=kwargs.get("epochs", 2000),
        weight_decay=kwargs.get("weight_decay", 1e-12),
        # Loss weights
        lw_X=kwargs.get("lw_X", 1.0),
        lw_E=kwargs.get("lw_E", 5.0),
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Logging
        verbose="progress_bar",
    )
    
    print(f"\nStarting training on {len(data['X_train'])} molecules...")
    
    # DiGress fit() only takes X_train (no y_train)
    model.fit(X_train=data["X_train"])
    
    # Save model
    checkpoint_path = output_dir / f"digress_{data['tranche']}_unconditional.pt"
    model.save_to_local(str(checkpoint_path))
    print(f"\nModel saved to {checkpoint_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train DiGress for unconditional molecular generation")
    parser.add_argument("--tranche", choices=["BBAB", "FBAB", "JBCD"], required=True,
                        help="ZINC tranche to train on")
    parser.add_argument("--data_dir", type=str,
                        default="experiments/zinc_baselines/data",
                        help="Directory containing prepared data")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/zinc_baselines/checkpoints",
                        help="Directory to save checkpoints")
    # Model hyperparameters (matched to GRASSY-DiT for fair comparison)
    parser.add_argument("--hidden_size_X", type=int, default=256,
                        help="Hidden dimension for node features")
    parser.add_argument("--hidden_size_E", type=int, default=128,
                        help="Hidden dimension for edge features")
    parser.add_argument("--num_layer", type=int, default=12)
    parser.add_argument("--n_head", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=500)
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2000)
    # Loss weights
    parser.add_argument("--lw_X", type=float, default=1.0,
                        help="Loss weight for node reconstruction")
    parser.add_argument("--lw_E", type=float, default=5.0,
                        help="Loss weight for edge reconstruction")
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    if not Path(args.data_dir).is_absolute():
        data_dir = PROJECT_ROOT / args.data_dir
    else:
        data_dir = Path(args.data_dir)
        
    if not Path(args.output_dir).is_absolute():
        output_dir = PROJECT_ROOT / args.output_dir
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"DiGress Training - {args.tranche} (unconditional)")
    print(f"{'='*60}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    print(f"\nLoading data for {args.tranche}...")
    data = load_data(args.tranche, data_dir)
    print(f"  Train: {len(data['X_train'])} molecules")
    print(f"  Val: {len(data['X_val'])} molecules")
    print(f"  Test: {len(data['X_test'])} molecules")
    
    kwargs = {
        "hidden_size_X": args.hidden_size_X,
        "hidden_size_E": args.hidden_size_E,
        "num_layer": args.num_layer,
        "n_head": args.n_head,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "lw_X": args.lw_X,
        "lw_E": args.lw_E,
    }
    
    train_unconditional(data, output_dir, **kwargs)
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
