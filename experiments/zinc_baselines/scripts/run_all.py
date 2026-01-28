"""
Master script to run the full baseline training pipeline.

Usage:
    # Run everything
    python experiments/zinc_baselines/scripts/run_all.py
    
    # Run specific steps
    python experiments/zinc_baselines/scripts/run_all.py --step prepare
    python experiments/zinc_baselines/scripts/run_all.py --step train
    python experiments/zinc_baselines/scripts/run_all.py --step evaluate
    python experiments/zinc_baselines/scripts/run_all.py --step aggregate
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "zinc_baselines" / "scripts"

TRANCHES = ["BBAB", "FBAB", "JBCD"]


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        return False
    return True


def step_prepare():
    """Step 1: Prepare data."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "prepare_data.py")]
    return run_command(cmd, "Data Preparation")


def step_train(epochs: int = 1000, batch_size: int = 64):
    """Step 2: Train all models."""
    success = True
    
    # Train GraphDIT (unconditional + conditional)
    for tranche in TRANCHES:
        # Unconditional
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "train_graph_dit.py"),
            "--tranche", tranche,
            "--mode", "unconditional",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
        ]
        if not run_command(cmd, f"GraphDIT Unconditional - {tranche}"):
            success = False
        
        # Conditional
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "train_graph_dit.py"),
            "--tranche", tranche,
            "--mode", "conditional",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
        ]
        if not run_command(cmd, f"GraphDIT Conditional - {tranche}"):
            success = False
    
    # Train DiGress (unconditional only)
    for tranche in TRANCHES:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "train_digress.py"),
            "--tranche", tranche,
            "--epochs", str(epochs),
            "--batch_size", str(batch_size),
        ]
        if not run_command(cmd, f"DiGress Unconditional - {tranche}"):
            success = False
    
    return success


def step_evaluate(n_samples: int = 10000):
    """Step 3: Evaluate all models."""
    success = True
    
    # Evaluate GraphDIT
    for tranche in TRANCHES:
        for mode in ["unconditional", "conditional"]:
            cmd = [
                sys.executable, str(SCRIPTS_DIR / "evaluate.py"),
                "--model", "graph_dit",
                "--tranche", tranche,
                "--mode", mode,
                "--n_samples", str(n_samples),
            ]
            if not run_command(cmd, f"Evaluate GraphDIT {mode} - {tranche}"):
                success = False
    
    # Evaluate DiGress (unconditional only)
    for tranche in TRANCHES:
        cmd = [
            sys.executable, str(SCRIPTS_DIR / "evaluate.py"),
            "--model", "digress",
            "--tranche", tranche,
            "--mode", "unconditional",
            "--n_samples", str(n_samples),
        ]
        if not run_command(cmd, f"Evaluate DiGress unconditional - {tranche}"):
            success = False
    
    return success


def step_aggregate():
    """Step 4: Aggregate results."""
    cmd = [sys.executable, str(SCRIPTS_DIR / "aggregate_results.py")]
    return run_command(cmd, "Aggregate Results")


def main():
    parser = argparse.ArgumentParser(description="Run full baseline training pipeline")
    parser.add_argument("--step", choices=["prepare", "train", "evaluate", "aggregate", "all"],
                        default="all", help="Which step to run")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--n_samples", type=int, default=10000, help="Samples for evaluation")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ZINC Baselines Training Pipeline")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Step: {args.step}")
    
    if args.step == "prepare" or args.step == "all":
        if not step_prepare():
            print("\nData preparation failed!")
            if args.step == "all":
                return
    
    if args.step == "train" or args.step == "all":
        if not step_train(args.epochs, args.batch_size):
            print("\nTraining had some failures!")
            if args.step == "all":
                print("Continuing to evaluation...")
    
    if args.step == "evaluate" or args.step == "all":
        if not step_evaluate(args.n_samples):
            print("\nEvaluation had some failures!")
    
    if args.step == "aggregate" or args.step == "all":
        step_aggregate()
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
