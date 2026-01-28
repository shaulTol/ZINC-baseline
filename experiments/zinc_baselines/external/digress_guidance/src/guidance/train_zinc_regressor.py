"""
Train property regressor for ZINC dataset.
Predicts qed, MolLogP, num_atoms from noised molecular graphs.

Usage:
    python -m src.guidance.train_zinc_regressor experiment=zinc_regressor dataset.tranche=BBAB
"""
from rdkit import Chem

import torch
import wandb
import hydra
import omegaconf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings

import src.utils as utils
from src.datasets.zinc_dataset import ZINCDataModule, ZINCinfos, get_train_smiles
from src.metrics.molecular_metrics import SamplingMolecularMetrics, TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.guidance.zinc_regressor_discrete import ZincRegressorDiscrete


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {
        'name': cfg.general.name, 
        'project': 'zinc_regressor', 
        'config': config_dict,
        'settings': wandb.Settings(_disable_stats=True),
        'reinit': True, 
        'mode': cfg.general.wandb
    }
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(version_base='1.1', config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    print(f"Dataset config: {dataset_config}")
    assert dataset_config["name"] == 'zinc', "This script is for ZINC dataset"
    assert cfg.model.type == 'discrete'
    
    # Initialize datamodule
    datamodule = ZINCDataModule(cfg, regressor=True)
    datamodule.prepare_data()  # Must be called before ZINCinfos (needs dataloaders)
    dataset_infos = ZINCinfos(datamodule=datamodule, cfg=cfg)
    train_smiles = None  # Not needed for regressor training

    # Extra features
    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule, 
        extra_features=extra_features,
        domain_features=domain_features
    )
    
    # Output dims for regression
    guidance_target = cfg.general.guidance_target
    if guidance_target == 'all':
        num_targets = 3  # qed, logp, num_atoms
    else:
        num_targets = 1
    dataset_infos.output_dims = {'X': 0, 'E': 0, 'y': num_targets}

    # Metrics
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(remove_h=True, dataset_infos=dataset_infos)

    model_kwargs = {
        'dataset_infos': dataset_infos, 
        'train_metrics': train_metrics,
        'sampling_metrics': sampling_metrics, 
        'visualization_tools': visualization_tools,
        'extra_features': extra_features, 
        'domain_features': domain_features
    }

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    model = ZincRegressorDiscrete(cfg=cfg, **model_kwargs)

    # Callbacks
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename='{epoch}',
            monitor='val/epoch_mae',
            save_last=True,
            save_top_k=3,
            mode='min',
            every_n_epochs=1
        )
        print(f"Checkpoints will be saved to {checkpoint_callback.dirpath}")
        callbacks.append(checkpoint_callback)

    # Trainer
    name = cfg.general.name
    if name == 'debug':
        print("[WARNING]: Run is called 'debug' -- will run with fast_dev_run.")
        
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        accelerator='gpu' if cfg.general.gpus > 0 and torch.cuda.is_available() else 'cpu',
        devices=1 if cfg.general.gpus > 0 and torch.cuda.is_available() else None,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == 'debug',
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=False  # Disable logger to avoid YAML serialization issues
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    
    print(f"\nTraining complete! Best val MAE: {model.best_val_mae:.4f}")


if __name__ == '__main__':
    main()
