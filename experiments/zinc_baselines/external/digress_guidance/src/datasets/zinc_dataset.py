"""
ZINC dataset for DiGress with property guidance.
Supports qed, MolLogP, num_atoms conditioning.
"""
import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED as QEDModule
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics


def files_exist(files) -> bool:
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


def compute_zinc_properties(mol):
    """Compute qed, MolLogP, num_atoms for a molecule."""
    try:
        qed = QEDModule.qed(mol)
        logp = Descriptors.MolLogP(mol)
        num_atoms = float(mol.GetNumHeavyAtoms())
        return torch.tensor([qed, logp, num_atoms], dtype=torch.float)
    except:
        return None


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectQEDTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]  # First column is qed
        return data


class SelectLogPTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:2]  # Second column is logp
        return data


class SelectNumAtomsTransform:
    def __call__(self, data):
        data.y = data.y[..., 2:3]  # Third column is num_atoms
        return data


class ZINCDataset(InMemoryDataset):
    """ZINC dataset for DiGress guidance."""
    
    def __init__(self, stage, root, tranche='BBAB', transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            stage: 'train', 'val', or 'test'
            root: data directory containing raw/data_bbab/, raw/data_fbab/, raw/data_jbcd/
            tranche: 'BBAB', 'FBAB', or 'JBCD'
        """
        self.stage = stage
        self.tranche = tranche.lower()
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return [f'data_{self.tranche}/train/molecules.csv',
                f'data_{self.tranche}/val/molecules.csv', 
                f'data_{self.tranche}/test/molecules.csv']

    @property
    def processed_file_names(self):
        return [f'zinc_{self.tranche}_train.pt', 
                f'zinc_{self.tranche}_val.pt', 
                f'zinc_{self.tranche}_test.pt']

    def download(self):
        # Data should already exist via symlinks
        pass

    def process(self):
        RDLogger.DisableLog('rdApp.*')
        
        # ZINC atom types (common heavy atoms)
        types = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'S': 4, 'Cl': 5, 'Br': 6, 'I': 7, 'P': 8}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
        
        for split_idx, split_name in enumerate(['train', 'val', 'test']):
            csv_path = osp.join(self.raw_dir, f'data_{self.tranche}', split_name, 'molecules.csv')
            
            if not osp.exists(csv_path):
                print(f"Warning: {csv_path} not found, skipping")
                continue
                
            df = pd.read_csv(csv_path)
            smiles_list = df['smiles'].tolist()
            
            data_list = []
            
            for smi in tqdm(smiles_list, desc=f'Processing {split_name}'):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue
                
                # Compute properties
                props = compute_zinc_properties(mol)
                if props is None:
                    continue
                
                N = mol.GetNumAtoms()
                
                # Node features (atom types)
                type_idx = []
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol in types:
                        type_idx.append(types[symbol])
                    else:
                        # Unknown atom type, skip molecule
                        type_idx = None
                        break
                
                if type_idx is None:
                    continue
                
                # One-hot encode atom types
                x = F.one_hot(torch.tensor(type_idx, dtype=torch.long), num_classes=len(types)).float()
                
                # Edge features
                row, col, edge_type = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    bt = bond.GetBondType()
                    edge_type += [bonds.get(bt, 0) + 1, bonds.get(bt, 0) + 1]  # +1 for no-bond as 0
                
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_type_tensor = torch.tensor(edge_type, dtype=torch.long)
                # One-hot encode edge types (5 classes: no-edge, single, double, triple, aromatic)
                edge_attr = F.one_hot(edge_type_tensor, num_classes=5).float()
                
                # Create data object
                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=props.unsqueeze(0),  # (1, 3) for qed, logp, num_atoms
                    smiles=smi
                )
                
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                    
                data_list.append(data)
            
            torch.save(self.collate(data_list), self.processed_paths[split_idx])
            print(f"Saved {len(data_list)} molecules to {self.processed_paths[split_idx]}")


class ZINCDataModule(MolecularDataModule):
    """DataModule for ZINC dataset."""
    
    def __init__(self, cfg, regressor=False):
        self.datadir = cfg.dataset.datadir
        self.tranche = cfg.dataset.tranche
        self.regressor = regressor
        super().__init__(cfg)
    
    def prepare_data(self) -> None:
        # Select property transform based on guidance target
        target = getattr(self.cfg.general, 'guidance_target', 'all')
        if self.regressor and target == 'qed':
            transform = SelectQEDTransform()
        elif self.regressor and target == 'logp':
            transform = SelectLogPTransform()
        elif self.regressor and target == 'num_atoms':
            transform = SelectNumAtomsTransform()
        elif self.regressor and target == 'all':
            transform = None  # Keep all 3 properties
        else:
            transform = RemoveYTransform()
        
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        
        datasets = {
            'train': ZINCDataset(stage='train', root=root_path, tranche=self.tranche, 
                                transform=transform if self.regressor else RemoveYTransform()),
            'val': ZINCDataset(stage='val', root=root_path, tranche=self.tranche,
                              transform=transform if self.regressor else RemoveYTransform()),
            'test': ZINCDataset(stage='test', root=root_path, tranche=self.tranche,
                               transform=transform)
        }
        super().prepare_data(datasets)


class ZINCinfos(AbstractDatasetInfos):
    """Dataset info for ZINC."""
    
    def __init__(self, datamodule, cfg):
        self.remove_h = cfg.dataset.remove_h
        self.name = 'zinc'
        self.n_nodes = datamodule.node_counts()
        self.node_types = datamodule.node_types()
        self.edge_types = datamodule.edge_counts()
        
        super().complete_infos(self.n_nodes, self.node_types)
        
        # ZINC atom decoder
        self.atom_decoder = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'P']
        self.num_atom_types = len(self.atom_decoder)
        self.valencies = [4, 3, 2, 1, 2, 1, 1, 1, 3]  # Typical valencies
        self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19, 4: 32, 5: 35, 6: 80, 7: 127, 8: 31}
        self.max_weight = int(self.max_n_nodes * max(self.atom_weights.values()))
        
        # Property info
        self.property_names = ['qed', 'logp', 'num_atoms']
        self.num_properties = 3
        
        # Valency distribution - compute from data
        self.valency_distribution = datamodule.valency_count(self.max_n_nodes)


def get_train_smiles(cfg, datamodule, dataset_infos):
    """Get training SMILES for evaluation metrics."""
    train_smiles = []
    for data in datamodule.train_dataloader():
        if hasattr(data, 'smiles'):
            if isinstance(data.smiles, list):
                train_smiles.extend(data.smiles)
            else:
                train_smiles.append(data.smiles)
    return train_smiles
