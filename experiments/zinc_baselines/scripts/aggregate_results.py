"""
Aggregate results into LaTeX-ready tables.

Usage:
    python experiments/zinc_baselines/scripts/aggregate_results.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

RESULTS_DIR = PROJECT_ROOT / "experiments" / "zinc_baselines" / "results"
OUTPUT_DIR = RESULTS_DIR

MODELS = ["graph_dit", "digress"]
TRANCHES = ["BBAB", "FBAB", "JBCD"]


def aggregate_unconditional():
    """Create Table 1: Unconditional generation metrics."""
    print("\n" + "="*60)
    print("TABLE 1: Unconditional Generation")
    print("="*60)
    
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
            else:
                print(f"  Missing: {csv_path}")
    
    if not rows:
        print("No unconditional results found!")
        return None
    
    results_df = pd.DataFrame(rows)
    
    # Select columns for LaTeX table
    latex_cols = ["tranche", "model", "validity", "uniqueness", "novelty", "diversity", "similarity", "fcd"]
    available_cols = [c for c in latex_cols if c in results_df.columns]
    latex_df = results_df[available_cols].copy()
    
    # Sort by tranche then model
    latex_df = latex_df.sort_values(["tranche", "model"])
    
    print("\nResults:")
    print(latex_df.to_string(index=False))
    
    print("\nLaTeX format:")
    print(latex_df.to_latex(index=False, float_format="%.2f"))
    
    output_path = OUTPUT_DIR / "table1_unconditional.csv"
    latex_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return latex_df


def aggregate_conditional():
    """Create Table 2: Conditional generation metrics."""
    print("\n" + "="*60)
    print("TABLE 2: Conditional Generation (GraphDIT only)")
    print("="*60)
    
    rows = []
    
    for tranche in TRANCHES:
        # Only GraphDIT supports conditional
        model = "graph_dit"
        csv_path = RESULTS_DIR / model / f"{tranche}_conditional_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            row = df.iloc[0].to_dict()
            row["model"] = model
            row["tranche"] = tranche
            rows.append(row)
        else:
            print(f"  Missing: {csv_path}")
    
    if not rows:
        print("No conditional results found!")
        return None
    
    results_df = pd.DataFrame(rows)
    
    # Select key metrics for the table
    key_cols = ["tranche", "model", "validity", "n_valid_for_props"]
    
    # Add MAE for each property
    property_names = ["qed", "HeavyAtomMolWt", "MolWt", "TPSA", "MolLogP", "num_atoms"]
    for prop in property_names:
        mae_col = f"{prop}_mae"
        if mae_col in results_df.columns:
            key_cols.append(mae_col)
    
    available_cols = [c for c in key_cols if c in results_df.columns]
    latex_df = results_df[available_cols].copy()
    
    print("\nResults:")
    print(latex_df.to_string(index=False))
    
    print("\nLaTeX format:")
    print(latex_df.to_latex(index=False, float_format="%.4f"))
    
    output_path = OUTPUT_DIR / "table2_conditional.csv"
    latex_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return latex_df


def main():
    print("="*60)
    print("Aggregating Results for LaTeX Tables")
    print("="*60)
    
    table1 = aggregate_unconditional()
    table2 = aggregate_conditional()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Output directory: {OUTPUT_DIR}")
    if table1 is not None:
        print(f"  - table1_unconditional.csv")
    if table2 is not None:
        print(f"  - table2_conditional.csv")


if __name__ == "__main__":
    main()
