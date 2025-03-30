from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch import nn
import lightning.pytorch as lp
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models.lstm import LSTM, LSTMConfig
from constants import input_dim


@dataclass
class CollisionPredictionConfig:
    net_config: LSTMConfig 
    learning_rate: float 
    net: LSTM
    lr_scheduler: Optional[str] = None
    lr_scheduler_params: dict = field(default_factory=dict)
    total_steps: Optional[int] = None  # Required for OneCycleLR


class CollisionPrediction(lp.LightningModule):
    def __init__(self, pipeline_config):
        super().__init__()
        self.pipeline_config = pipeline_config
        self.net = self.pipeline_config.net(
            net_config=self.pipeline_config.net_config
        )
        self.learning_rate = self.pipeline_config.learning_rate
        self.lr_scheduler = self.pipeline_config.lr_scheduler
        self.lr_scheduler_params = self.pipeline_config.lr_scheduler_params
        self.total_steps = self.pipeline_config.total_steps
        self.criterion = nn.BCELoss()
        # Add steps_ahead attribute if it doesn't exist (for APT calculation)
        if not hasattr(self, 'steps_ahead'):
            self.steps_ahead = 1  # Default value, should be set by the user

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Log training loss with 'train' prefix
        self.log('train/loss', loss, 
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Predict and calculate metrics
        predicted = (y_hat > 0.5).float()
        y_true = y.cpu().numpy().flatten().tolist()
        y_pred = predicted.cpu().numpy().flatten().tolist()

        # Calculate validation metrics with safe handling
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle cases where metrics may be undefined
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            self.log('val/metric_error', 1.0, sync_dist=True)
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        # Log validation metrics with 'val' prefix
        self.log('val/loss', loss, sync_dist=True)
        self.log('val/accuracy', accuracy, sync_dist=True)
        self.log('val/precision', precision, sync_dist=True)
        self.log('val/recall', recall, sync_dist=True)
        self.log('val/f1', f1, sync_dist=True)

        return {
            'val_loss': loss,
            'val_accuracy': accuracy,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        predicted = (y_hat > 0.5).float()
        y_true = y.cpu().numpy().flatten().tolist()
        y_pred = predicted.cpu().numpy().flatten().tolist()

        # Calculate test metrics with safe handling
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle cases where metrics may be undefined using zero_division parameter
        try:
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
        except Exception as e:
            self.log('test/metric_error', 1.0, sync_dist=True)
            precision, recall, f1 = 0.0, 0.0, 0.0

        # True Positives (TP) and False Positives (FP)
        TP = sum([(1 if (ypred == ytrue == 1) else 0) for ypred, ytrue in zip(y_pred, y_true)])
        FP = sum([(1 if (ypred == 1 and ytrue == 0) else 0) for ypred, ytrue in zip(y_pred, y_true)])

        # Collision Prediction metrics with safe division
        predicted_collisions = sum(y_pred)
        total_collisions = sum(y_true)
        
        # Handle potential zero division cases
        CPP = predicted_collisions / total_collisions if total_collisions else 0
        CDP = predicted_collisions / (FP + total_collisions) if (FP + total_collisions) else 0
        
        # Average Prediction Time (APT) with safe division
        average_prediction_time = self.steps_ahead * predicted_collisions
        APT = average_prediction_time / total_collisions if total_collisions else 0

        # Log test metrics with 'test' prefix
        self.log('test/accuracy', accuracy, sync_dist=True)
        self.log('test/precision', precision, sync_dist=True)
        self.log('test/recall', recall, sync_dist=True)
        self.log('test/f1', f1, sync_dist=True)
        self.log('test/cpp', CPP, sync_dist=True)
        self.log('test/cdp', CDP, sync_dist=True)
        self.log('test/apt', APT, sync_dist=True)
        
        # Log additional information about the data distribution
        self.log('test/total_predictions', len(y_pred), sync_dist=True)
        self.log('test/positive_predictions', predicted_collisions, sync_dist=True)
        self.log('test/actual_positives', total_collisions, sync_dist=True)

        return {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_cpp': CPP,
            'test_cdp': CDP,
            'test_apt': APT
        }

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # If no scheduler is specified, return just the optimizer
        if not self.lr_scheduler:
            return optimizer

        # OneCycleLR scheduler
        if self.lr_scheduler == 'onecycle':
            from torch.optim.lr_scheduler import OneCycleLR
            
            # Default parameters if not provided
            default_params = {
                'max_lr': self.learning_rate * 3,
                'total_steps': self.total_steps,
                'pct_start': 0.3,
                'div_factor': 25,
                'final_div_factor': 1e4
            }
            
            # Merge default params with user-provided params
            scheduler_params = {**default_params, **self.lr_scheduler_params}
            
            # Validate total_steps is provided for OneCycleLR
            if scheduler_params.get('total_steps') is None:
                raise ValueError("total_steps must be provided for OneCycleLR scheduler")
            
            scheduler = OneCycleLR(
                optimizer, 
                **scheduler_params
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        # Cosine annealing with warmup
        elif self.lr_scheduler == 'cosine_warmup':
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            
            # Default parameters if not provided
            default_params = {
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 1e-6
            }
            
            # Merge default params with user-provided params
            scheduler_params = {**default_params, **self.lr_scheduler_params}
            
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, 
                **scheduler_params
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        # Step scheduler
        elif self.lr_scheduler == 'step':
            from torch.optim.lr_scheduler import StepLR
            
            # Default parameters if not provided
            default_params = {
                'step_size': 10,
                'gamma': 0.1
            }
            
            # Merge default params with user-provided params
            scheduler_params = {**default_params, **self.lr_scheduler_params}
            
            scheduler = StepLR(optimizer, **scheduler_params)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        # Exponential scheduler
        elif self.lr_scheduler == 'exponential':
            from torch.optim.lr_scheduler import ExponentialLR
            
            # Default parameters if not provided
            default_params = {
                'gamma': 0.95
            }
            
            # Merge default params with user-provided params
            scheduler_params = {**default_params, **self.lr_scheduler_params}
            
            scheduler = ExponentialLR(optimizer, **scheduler_params)
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        # If an unsupported scheduler is specified
        raise ValueError(f"Unsupported learning rate scheduler: {self.lr_scheduler}")
        

#================================ Data Module ================================#


class CollisionDataModule(lp.LightningDataModule):
    def __init__(self, train_loader, val_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
