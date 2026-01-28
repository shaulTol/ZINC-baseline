"""
Property regressor for ZINC dataset.
Predicts qed, MolLogP, num_atoms from noised molecular graphs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
import src.utils as utils


def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()


class ZincRegressorDiscrete(pl.LightningModule):
    """Property regressor for ZINC molecules."""
    
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, 
                 extra_features, domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.args = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps
        )

        # Marginal transition model
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution: {x_marginals} for nodes, {e_marginals} for edges")
        
        self.transition_model = MarginalUniformTransition(
            x_marginals=x_marginals, 
            e_marginals=e_marginals,
            y_classes=self.ydim_output
        )

        self.limit_dist = utils.PlaceHolder(
            X=x_marginals, 
            E=e_marginals,
            y=torch.ones(self.ydim_output) / self.ydim_output
        )

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_mae = 1e8

        # Loss metrics
        self.train_loss = MeanSquaredError(squared=True)
        self.val_loss = MeanAbsoluteError()
        self.test_loss = MeanAbsoluteError()

        # Per-property metrics
        num_targets = self.ydim_output
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.val_loss_each = [MeanAbsoluteError().to(device) for _ in range(num_targets)]
        self.test_loss_each = [MeanAbsoluteError().to(device) for _ in range(num_targets)]
        
        # Property names for logging
        self.target_dict = {0: "qed", 1: "logp", 2: "num_atoms"}

    def training_step(self, data, i):
        # Store target, zero out y for noising
        target = data.y.clone()
        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        mse = self.compute_train_loss(pred, target, log=i % self.log_every_steps == 0)
        return {'loss': mse}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.args.train.lr, 
            amsgrad=True, 
            weight_decay=1e-12
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of input features:", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        train_mse = self.train_loss.compute()
        to_log = {"train_epoch/mse": train_mse}
        print(f"Epoch {self.current_epoch}: train_mse: {train_mse:.4f} -- {time.time() - self.start_epoch_time:.1f}s")
        wandb.log(to_log)
        self.train_loss.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        reset_metrics(self.val_loss_each)

    def validation_step(self, data, i):
        target = data.y.clone()
        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        mae = self.compute_val_loss(pred, target)
        return {'val_loss': mae}

    def validation_epoch_end(self, outs) -> None:
        val_mae = self.val_loss.compute()
        to_log = {"val/epoch_mae": val_mae}
        print(f"Epoch {self.current_epoch}: val_mae: {val_mae:.4f}")
        wandb.log(to_log)
        self.log('val/epoch_mae', val_mae, on_epoch=True, on_step=False)

        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
        print(f'Val loss: {val_mae:.4f} \t Best val loss: {self.best_val_mae:.4f}\n')

        # Log per-property MAE
        num_targets = min(len(self.val_loss_each), self.ydim_output)
        for i in range(num_targets):
            if i in self.target_dict:
                mae_each = self.val_loss_each[i].compute()
                print(f"  {self.target_dict[i]}: val_mae = {mae_each:.4f}")
                wandb.log({f"val_epoch/{self.target_dict[i]}_mae": mae_each})

        self.val_loss.reset()
        reset_metrics(self.val_loss_each)

    def on_test_epoch_start(self) -> None:
        self.test_loss.reset()
        reset_metrics(self.test_loss_each)

    def apply_noise(self, X, E, y, node_mask):
        """Sample noise and apply it to the data."""
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all()
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        probX = X @ Qtb.X
        probE = E @ Qtb.E.unsqueeze(1)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            't_int': t_int, 't': t_float, 'beta_t': beta_t, 
            'alpha_s_bar': alpha_s_bar, 'alpha_t_bar': alpha_t_bar, 
            'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask
        }
        return noisy_data

    def compute_val_loss(self, pred, target):
        """Compute MAE for validation."""
        num_targets = min(pred.y.shape[1], target.shape[1], len(self.val_loss_each))
        for i in range(num_targets):
            self.val_loss_each[i](pred.y[:, i], target[:, i])

        mae = self.val_loss(pred.y[:, :num_targets], target[:, :num_targets])
        return mae

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    def compute_extra_data(self, noisy_data):
        """Compute extra features for network input."""
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)

        t = noisy_data['t']
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=t)

    def compute_train_loss(self, pred, target, log: bool):
        """Compute MSE for training."""
        num_targets = min(pred.y.shape[1], target.shape[1])
        mse = self.train_loss(pred.y[:, :num_targets], target[:, :num_targets])

        if log:
            wandb.log({"train_loss/batch_mse": mse.item()}, commit=True)
        return mse
