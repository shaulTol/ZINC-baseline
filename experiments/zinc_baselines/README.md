# ZINC Baseline Training: GraphDIT & DiGress

Training scripts for Graph DiT and DiGress baselines on ZINC tranches (BBAB, FBAB, JBCD) for comparison with GRASSY-DiT.

## Quick Start

### 1. Prepare Data
Computes molecular properties (QED, MolWt, TPSA, MolLogP, etc.) for all tranches:
```bash
python experiments/zinc_baselines/scripts/prepare_data.py
```

### 2. Train Models

**GraphDIT (unconditional + conditional):**
```bash
# Unconditional
python experiments/zinc_baselines/scripts/train_graph_dit.py --tranche BBAB --mode unconditional
python experiments/zinc_baselines/scripts/train_graph_dit.py --tranche FBAB --mode unconditional
python experiments/zinc_baselines/scripts/train_graph_dit.py --tranche JBCD --mode unconditional

# Conditional (multi-property)
python experiments/zinc_baselines/scripts/train_graph_dit.py --tranche BBAB --mode conditional
python experiments/zinc_baselines/scripts/train_graph_dit.py --tranche FBAB --mode conditional
python experiments/zinc_baselines/scripts/train_graph_dit.py --tranche JBCD --mode conditional
```

**DiGress (unconditional only):**
```bash
python experiments/zinc_baselines/scripts/train_digress.py --tranche BBAB
python experiments/zinc_baselines/scripts/train_digress.py --tranche FBAB
python experiments/zinc_baselines/scripts/train_digress.py --tranche JBCD
```

### 3. Evaluate
```bash
# GraphDIT
python experiments/zinc_baselines/scripts/evaluate.py --model graph_dit --tranche BBAB --mode unconditional
python experiments/zinc_baselines/scripts/evaluate.py --model graph_dit --tranche BBAB --mode conditional

# DiGress
python experiments/zinc_baselines/scripts/evaluate.py --model digress --tranche BBAB --mode unconditional
```

### 4. Aggregate Results
```bash
python experiments/zinc_baselines/scripts/aggregate_results.py
```

## Hyperparameters

Matched to GRASSY-DiT for fair comparison:

| Parameter | Value |
|-----------|-------|
| Hidden size | 256 |
| Layers | 12 |
| Heads | 16 |
| Timesteps | 500 |
| Batch size | 16 |
| Epochs | 2000 |
| Learning rate | 1e-4 |
| Guidance scale | 2.0 |
| Condition dropout | 0.1 |

## Conditioning Properties

For conditional generation (GraphDIT only):
- `qed` - Quantitative Estimate of Drug-likeness
- `HeavyAtomMolWt` - Heavy Atom Molecular Weight
- `MolWt` - Molecular Weight
- `TPSA` - Topological Polar Surface Area
- `MolLogP` - Partition Coefficient
- `num_atoms` - Number of Heavy Atoms

## Output

Results are saved to `experiments/zinc_baselines/results/`:

- **Table 1** (Unconditional): Validity, Uniqueness, Novelty, Diversity, Similarity, FCD
- **Table 2** (Conditional): Property MAE for each conditioning property (GraphDIT only)

## Project Structure

```
experiments/zinc_baselines/
├── data/                    # Prepared .pkl files with properties
├── checkpoints/             # Trained model weights
├── results/
│   ├── graph_dit/          # GraphDIT evaluation results
│   └── digress/            # DiGress evaluation results
├── scripts/
│   ├── prepare_data.py     # Step 1: Compute properties
│   ├── train_graph_dit.py  # Step 2a: Train GraphDIT
│   ├── train_digress.py    # Step 2b: Train DiGress
│   ├── evaluate.py         # Step 3: Generate & evaluate
│   ├── aggregate_results.py # Step 4: Create tables
│   └── run_all.py          # Run full pipeline
└── README.md
```

## Notes

- **DiGress**: Only supports unconditional generation in torch-molecule
- **GraphDIT**: Supports both unconditional and conditional (multi-property) generation
- Data must be pre-split into `data/data_bbab/`, `data/data_fbab/`, `data/data_jbcd/` folders
