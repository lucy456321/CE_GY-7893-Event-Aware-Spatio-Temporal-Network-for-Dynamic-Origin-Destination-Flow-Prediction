"""
Training pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler
from torch.serialization import add_safe_globals
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import logging
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import warnings
import gc
import os
import scipy.sparse

warnings.filterwarnings('ignore')

from model import EASTGNNModel, create_adjacency_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. DATASET

class ODEventDataset(Dataset):
    """Dataset for OD flows with event features"""
    def __init__(self, data_path: str, time_window: int = 8, prediction_horizon: int = 4):
        super().__init__()
        self.time_window = time_window
        self.prediction_horizon = prediction_horizon

        # Load data 
        data = torch.load(data_path, map_location='cpu', weights_only=False)

        self.od_tensor = data['od_tensor']
        self.event_tensor = data['event_tensor']

        # Calculate number of sequential samples
        self.num_samples = len(self.od_tensor) - time_window - prediction_horizon + 1

        # Free up the dict 
        del data

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hist_end = idx + self.time_window
        pred_end = hist_end + self.prediction_horizon

        return {
            'historical_od': self.od_tensor[idx:hist_end],
            'event_features': self.event_tensor[idx:hist_end],
            'target_od': self.od_tensor[hist_end:pred_end],
        }

#2. LOSS FUNCTION 

class SurgeAwareLoss(nn.Module):
    def __init__(self, surge_threshold: float = 2.0, surge_weight: float = 3.0, quantile: float = 0.9):
        super().__init__()
        self.surge_threshold = surge_threshold
        self.surge_weight = surge_weight
        self.quantile = quantile

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Base MSE
        mse_loss = F.mse_loss(predictions, targets, reduction='none')

        # 2. Surge Weighted MSE
        batch_mean = targets.mean()
        batch_std = targets.std()

        # Create weights tensor
        weights = torch.ones_like(targets)
        weights[targets > (batch_mean + self.surge_threshold * batch_std)] = self.surge_weight

        weighted_mse = (weights * mse_loss).mean()

        # 3. MAE
        mae_loss = F.l1_loss(predictions, targets)

        # 4. Quantile Loss 
        errors = targets - predictions
        quantile_loss = torch.mean(torch.max(self.quantile * errors, (self.quantile - 1) * errors))

        # 5. Marginal Losses
        origin_loss = F.mse_loss(predictions.sum(dim=-1), targets.sum(dim=-1))
        dest_loss = F.mse_loss(predictions.sum(dim=-2), targets.sum(dim=-2))

        total_loss = (
            weighted_mse + 
            0.5 * mae_loss + 
            0.2 * quantile_loss + 
            0.1 * origin_loss + 
            0.1 * dest_loss
        )

        return {'total': total_loss, 'mae': mae_loss}

#3. TRAINER

class Trainer:
    def __init__(self, model, train_loader, val_loader, adj_matrix, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.adj_matrix = adj_matrix.to(device)
        self.config = config
        self.device = device

        # Detect device type for AMP
        self.device_type = 'cuda' if 'cuda' in str(device) else 'cpu'
        self.scaler = torch.amp.GradScaler('cuda') if self.device_type == 'cuda' else None

        self.criterion = SurgeAwareLoss(
            surge_threshold=config['surge_threshold'],
            surge_weight=config['surge_weight'],
            quantile=config['quantile']
        )

        self.optimizer = Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.metrics_history = []

        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', leave=False)

        for batch in pbar:
            historical_od = batch['historical_od'].to(self.device, non_blocking=True)
            event_features = batch['event_features'].to(self.device, non_blocking=True)
            target_od = batch['target_od'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Context
            with torch.amp.autocast(device_type=self.device_type):
                predictions = self.model(historical_od, event_features, self.adj_matrix)
                loss_dict = self.criterion(predictions, target_od)
                loss = loss_dict['total']

            # Scaled Backward Pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': loss.item()})

        return {'train_loss': total_loss / num_batches}

    @torch.inference_mode() 
    def validate(self) -> Dict[str, float]:
        """Streaming Validation (Low RAM)"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        total_samples = 0

        for batch in tqdm(self.val_loader, desc='Validation', leave=False):
            historical_od = batch['historical_od'].to(self.device, non_blocking=True)
            event_features = batch['event_features'].to(self.device, non_blocking=True)
            target_od = batch['target_od'].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device_type):
                predictions = self.model(historical_od, event_features, self.adj_matrix)
                loss_dict = self.criterion(predictions, target_od)

            # Accumulate metrics
            batch_size = target_od.size(0)
            total_loss += loss_dict['total'].item() * batch_size

            diff = (predictions - target_od).detach().float()
            total_mae += torch.abs(diff).mean().item() * batch_size
            total_mse += torch.pow(diff, 2).mean().item() * batch_size

            total_samples += batch_size

        return {
            'val_loss': total_loss / total_samples,
            'MAE': total_mae / total_samples,
            'RMSE': np.sqrt(total_mse / total_samples)
        }

    def train(self, num_epochs: int):
        logger.info(f"STARTING TRAINING - {num_epochs} epochs | Device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics['train_loss'])

            val_metrics = self.validate()
            self.val_losses.append(val_metrics['val_loss'])
            self.metrics_history.append({**train_metrics, **val_metrics})

            self.scheduler.step(val_metrics['val_loss'])

            logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | MAE: {val_metrics['MAE']:.4f}")

            # Save best
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, val_metrics, is_best=True)

            # Periodic save & Cleanup
            if epoch % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)
                gc.collect() 
                if self.device_type == 'cuda':
                    torch.cuda.empty_cache()

        self.save_metrics()
        self.plot_training_curves()

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        path = self.checkpoint_dir / ('best_model.pt' if is_best else f'ckpt_ep{epoch}.pt')
        torch.save(checkpoint, path)

    def save_metrics(self):
        with open(self.log_dir / 'training_metrics.json', 'w') as f:
            json.dump({'history': self.metrics_history}, f, indent=2, default=str)

    def plot_training_curves(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(self.log_dir / 'loss_curve.png')
        plt.close()

# 4. MAIN UTILS

def register_pandas_global():
    try:
        add_safe_globals({'pandas.core.indexes.datetimes._new_DatetimeIndex': 'pandas.core.indexes.datetimes._new_DatetimeIndex'})
    except:
        pass

def main():
    # 1. Configuration
    optimal_workers = 0 

    config = {
        'data_dir': './processed_data', 'time_window': 8, 'prediction_horizon': 4,
        'num_zones': 263, 'event_feature_dim': 22, 'hidden_dim': 64,
        'num_gnn_layers': 2, 'num_temporal_layers': 2, 'dropout': 0.1,
        'batch_size': 8, 
        'num_epochs': 50, 'learning_rate': 1e-3, 'weight_decay': 1e-5, 'grad_clip': 5.0,
        'surge_threshold': 2.0, 'surge_weight': 3.0, 'quantile': 0.9,
        'checkpoint_dir': './checkpoints', 'log_dir': './logs',
        'save_freq': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': optimal_workers
    }

    # 2. Setup
    register_pandas_global()
    device = torch.device(config['device'])

    # 3. Data Loading
    train_dataset = ODEventDataset(f"{config['data_dir']}/train_data.pt", config['time_window'], config['prediction_horizon'])
    val_dataset = ODEventDataset(f"{config['data_dir']}/val_data.pt", config['time_window'], config['prediction_horizon'])

    # DISABLED
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, 
        num_workers=config['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'], shuffle=False, 
        num_workers=config['num_workers'], pin_memory=True
    )

    # 4. Model & Train
    adj_matrix = create_adjacency_matrix(config['num_zones'], k=8)
    model = EASTGNNModel(
        num_zones=config['num_zones'], event_feature_dim=config['event_feature_dim'],
        hidden_dim=config['hidden_dim'], num_gnn_layers=config['num_gnn_layers'],
        num_temporal_layers=config['num_temporal_layers'], output_horizons=config['prediction_horizon'],
        dropout=config['dropout']
    )

    try:
        model = torch.compile(model)
        logger.info("Model compiled with torch.compile() for speed.")
    except Exception as e:
        logger.warning(f"Could not compile model (safe to ignore): {e}")

    trainer = Trainer(model, train_loader, val_loader, adj_matrix, config, device)
    trainer.train(config['num_epochs'])

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium') 
    main()
