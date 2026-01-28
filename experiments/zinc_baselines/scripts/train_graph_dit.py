"""
Train GraphDIT for unconditional and conditional molecular generation.

Usage:
    # Unconditional
    python experiments/zinc_baselines/scripts/train_graph_dit.py --tranche BBAB --mode unconditional
    
    # Conditional (multi-property)
    python experiments/zinc_baselines/scripts/train_graph_dit.py --tranche BBAB --mode conditional

Note: Class is GraphDITMolecularGenerator (capital IT), not GraphDiTMolecularGenerator
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

from torch_molecule.generator.graph_dit import GraphDITMolecularGenerator


def load_data(tranche: str, data_dir: Path) -> dict:
    """Load prepared data for a tranche."""
    data_path = data_dir / f"{tranche}_prepared.pkl"
    with open(data_path, "rb") as f:
        return pickle.load(f)


def train_unconditional(data: dict, output_dir: Path, checkpoint_dir: Path = None,
                        checkpoint_freq: int = 100, resume_from: str = None, **kwargs):
    """Train GraphDIT for unconditional generation."""
    
    print(f"\nInitializing GraphDIT for unconditional generation...")
    print(f"  hidden_size: {kwargs.get('hidden_size', 256)}")
    print(f"  num_layer: {kwargs.get('num_layer', 12)}")
    print(f"  epochs: {kwargs.get('epochs', 2000)}")
    print(f"  batch_size: {kwargs.get('batch_size', 16)}")
    if checkpoint_dir:
        print(f"  checkpoint_dir: {checkpoint_dir}")
        print(f"  checkpoint_freq: {checkpoint_freq}")
    if resume_from:
        print(f"  resume_from: {resume_from}")
    
    model = GraphDITMolecularGenerator(
        # Model architecture (matched to GRASSY-DiT)
        hidden_size=kwargs.get("hidden_size", 256),
        num_layer=kwargs.get("num_layer", 12),
        num_head=kwargs.get("num_head", 16),
        dropout=kwargs.get("dropout", 0.0),
        # Diffusion settings
        timesteps=kwargs.get("timesteps", 500),
        # Training settings
        batch_size=kwargs.get("batch_size", 16),
        learning_rate=kwargs.get("lr", 1e-4),
        epochs=kwargs.get("epochs", 2000),
        # No conditioning for unconditional
        task_type=[],
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Logging
        verbose="progress_bar",
    )
    
    print(f"\nStarting training on {len(data['X_train'])} molecules...")
    
    # For unconditional generation, pass y=None
    model.fit(
        X_train=data["X_train"],
        y_train=None,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        checkpoint_freq=checkpoint_freq,
        resume_from=resume_from,
    )
    
    # Save model
    checkpoint_path = output_dir / f"graph_dit_{data['tranche']}_unconditional.pt"
    model.save_to_local(str(checkpoint_path))
    print(f"\nModel saved to {checkpoint_path}")
    
    return model


def train_conditional(data: dict, output_dir: Path, checkpoint_dir: Path = None,
                      checkpoint_freq: int = 100, resume_from: str = None, **kwargs):
    """Train GraphDIT for multi-property conditional generation."""
    
    num_properties = len(data["property_names"])
    
    print(f"\nInitializing GraphDIT for conditional generation...")
    print(f"  Conditioning on {num_properties} properties: {data['property_names']}")
    print(f"  hidden_size: {kwargs.get('hidden_size', 256)}")
    print(f"  num_layer: {kwargs.get('num_layer', 12)}")
    print(f"  epochs: {kwargs.get('epochs', 2000)}")
    print(f"  batch_size: {kwargs.get('batch_size', 16)}")
    if checkpoint_dir:
        print(f"  checkpoint_dir: {checkpoint_dir}")
        print(f"  checkpoint_freq: {checkpoint_freq}")
    if resume_from:
        print(f"  resume_from: {resume_from}")
    
    # All properties are regression tasks
    task_type = ["regression"] * num_properties
    
    model = GraphDITMolecularGenerator(
        # Model architecture (matched to GRASSY-DiT)
        hidden_size=kwargs.get("hidden_size", 256),
        num_layer=kwargs.get("num_layer", 12),
        num_head=kwargs.get("num_head", 16),
        dropout=kwargs.get("dropout", 0.0),
        drop_condition=kwargs.get("drop_condition", 0.1),  # For classifier-free guidance
        # Conditioning
        task_type=task_type,
        # Diffusion settings
        timesteps=kwargs.get("timesteps", 500),
        # Sampling
        guide_scale=kwargs.get("guide_scale", 2.0),
        # Training settings
        batch_size=kwargs.get("batch_size", 16),
        learning_rate=kwargs.get("lr", 1e-4),
        epochs=kwargs.get("epochs", 2000),
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Logging
        verbose="progress_bar",
    )
    
    # Normalize properties using training set statistics
    y_train_normalized = (data["y_train"] - data["property_means"]) / (data["property_stds"] + 1e-8)
    
    # Handle NaN values - replace with 0 (mean after normalization)
    y_train_normalized = np.nan_to_num(y_train_normalized, nan=0.0)
    
    print(f"\nStarting training on {len(data['X_train'])} molecules...")
    
    model.fit(
        X_train=data["X_train"],
        y_train=y_train_normalized,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        checkpoint_freq=checkpoint_freq,
        resume_from=resume_from,
    )
    
    # Save model
    checkpoint_path = output_dir / f"graph_dit_{data['tranche']}_conditional.pt"
    model.save_to_local(str(checkpoint_path))
    
    # Save normalization statistics for inference
    stats_path = output_dir / f"graph_dit_{data['tranche']}_conditional_stats.pkl"
    with open(stats_path, "wb") as f:
        pickle.dump({
            "property_names": data["property_names"],
            "property_means": data["property_means"],
            "property_stds": data["property_stds"],
        }, f)
    
    print(f"\nModel saved to {checkpoint_path}")
    print(f"Stats saved to {stats_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train GraphDIT for molecular generation")
    parser.add_argument("--tranche", choices=["BBAB", "FBAB", "JBCD"], required=True,
                        help="ZINC tranche to train on")
    parser.add_argument("--mode", choices=["unconditional", "conditional"], required=True,
                        help="Training mode")
    parser.add_argument("--data_dir", type=str, 
                        default="experiments/zinc_baselines/data",
                        help="Directory containing prepared data")
    parser.add_argument("--output_dir", type=str,
                        default="experiments/zinc_baselines/checkpoints",
                        help="Directory to save final checkpoints")
    # Model hyperparameters (matched to GRASSY-DiT for fair comparison)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layer", type=int, default=12)
    parser.add_argument("--num_head", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=500)
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2000)
    # Conditional-specific
    parser.add_argument("--drop_condition", type=float, default=0.1,
                        help="Dropout rate for condition embedding (classifier-free guidance)")
    parser.add_argument("--guide_scale", type=float, default=2.0,
                        help="Guidance scale for conditional generation")
    # Checkpoint options
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Directory to save intermediate training checkpoints (optional)")
    parser.add_argument("--checkpoint_freq", type=int, default=100,
                        help="Save checkpoint every N epochs (default: 100)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to training checkpoint to resume from (optional)")
    
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
    print(f"GraphDIT Training - {args.tranche} ({args.mode})")
    print(f"{'='*60}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    print(f"\nLoading data for {args.tranche}...")
    data = load_data(args.tranche, data_dir)
    print(f"  Train: {len(data['X_train'])} molecules")
    print(f"  Val: {len(data['X_val'])} molecules")
    print(f"  Test: {len(data['X_test'])} molecules")
    
    kwargs = {
        "hidden_size": args.hidden_size,
        "num_layer": args.num_layer,
        "num_head": args.num_head,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "drop_condition": args.drop_condition,
        "guide_scale": args.guide_scale,
    }
    
    # Handle checkpoint directory
    checkpoint_dir = None
    if args.checkpoint_dir is not None:
        if not Path(args.checkpoint_dir).is_absolute():
            checkpoint_dir = PROJECT_ROOT / args.checkpoint_dir
        else:
            checkpoint_dir = Path(args.checkpoint_dir)
    
    if args.mode == "unconditional":
        train_unconditional(
            data, output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=args.checkpoint_freq,
            resume_from=args.resume_from,
            **kwargs
        )
    else:
        train_conditional(
            data, output_dir,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=args.checkpoint_freq,
            resume_from=args.resume_from,
            **kwargs
        )
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
