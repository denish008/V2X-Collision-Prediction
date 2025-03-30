import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from multiprocessing import cpu_count
import numpy as np
from glob import glob
from typing import List, Union, Literal

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as lp
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
from loguru import logger

from models.lstm import LSTM, LSTMConfig
from pipelines.experiment import Experiment
from pipelines.collision_prediction_pipeline import (
    CollisionDataModule, 
    CollisionPrediction, 
    CollisionPredictionConfig
)
from datasets.veins_dataset import VeinsDatasetConfig, VeinsDataset
from constants import column_headers


#====================================================================================#

# Experiment Details
experiment = Experiment(
    project="motorcycle_collision_prediction",
    model_name="lstm_12layer_256hiddim",
    run_num="run2",
    notes="""Using LSTM on Scenario A""",
)

#==================================Preparing Dataset==================================#

# Dataset
assets_dirpath = "assets"
assets_filepath = glob(
    os.path.join(assets_dirpath, "ScenarioA", "*.csv")
)
dataset_config = VeinsDatasetConfig(
    csv_paths=assets_filepath,
    column_headers=column_headers,
    timesteps=20,
    seed=42
)
dataset = VeinsDataset(config=dataset_config)
train_dataset = dataset._get_dataset("train")
val_dataset = dataset._get_dataset("val")
test_dataset = dataset._get_dataset("test")

# Dataloaders 
batch_size = 128
num_workers = int(cpu_count() * 0.25)
persistent_workers = False
prefetch_factor = 2
pin_memory = True

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    pin_memory=pin_memory
)
logger.info(f"created train dataloader, batches: {len(train_loader):_}")
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    pin_memory=pin_memory
)
logger.info(f"created val dataloader, batches: {len(val_loader):_}")
test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor,
    pin_memory=pin_memory
)
logger.info(f"created test dataloader, batches: {len(test_loader):_}")


#====================================================================================#


def run(experiment, mode: Literal["train", "debug", "predict"]):
    """
    Instantiates the training 
    """
    #--------------- Trainer:callbacks ---------------# 
    
    model_summary = RichModelSummary(max_depth=3)
    progbar = RichProgressBar(refresh_rate=50)
    trainer_callbacks = [model_summary, progbar]
    
    if mode == "train":
        model_ckpt = ModelCheckpoint(
            dirpath=experiment.model_checkpoints_dirpath,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_top_k=-1,
            save_weights_only=True,
            save_last=True,
            auto_insert_metric_name=False,
            filename="epoch_{epoch}-step_{step}",
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        trainer_callbacks += [model_ckpt, lr_monitor]
    elif mode == "debug":
        logger.info("trainer callbacks are disabled")

    #---------------- Trainer:loggers ----------------# 
    
    trainer_loggers = []
    if mode == "train":
        wandb_logger = WandbLogger(
            entity="v2x-collision-project",
            project=experiment.project,
            name=experiment.name,
            notes=experiment.notes,
            version=experiment.name,
            resume="never",
        )
        csv_logger = CSVLogger(
            save_dir=experiment.csv_logger_dirpath,
            name=experiment.name,
            version=experiment.run_num,
        )
        # git_tag_logger = GitTagLogger(
        #     repo_path=os.path.join(
        #         user_dirpath,
        #         <repo_path>,
        #     ),
        #     tag_name=experiment.name,
        #     message=experiment.notes,
        # )
        trainer_loggers += [wandb_logger, csv_logger]

    max_epochs = 100
    total_steps = (len(train_loader) * max_epochs)
    trainer = lp.Trainer(
        max_epochs=max_epochs,
        precision="32",
        accelerator="gpu",
        devices="auto",
        logger=trainer_loggers,
        callbacks=trainer_callbacks,
        num_sanity_val_steps=4,
        benchmark=True,
        enable_checkpointing=(False if mode == "debug" else None),
        log_every_n_steps=100,
        sync_batchnorm=True,
        strategy=DDPStrategy(find_unused_parameters=False),
        detect_anomaly=False
    )

    pipeline_config = CollisionPredictionConfig(
        net=LSTM,
        net_config=LSTMConfig(
            input_dim=9,
            hidden_dim=256, # 64 
            num_layers=12,  # 3
            output_dim=1,
            dropout_prob=0.3
        ),    
        learning_rate=1e-4,
        lr_scheduler="onecycle",
        total_steps=total_steps
    )
    model = CollisionPrediction(
        pipeline_config=pipeline_config
    )

    checkpoint_filepath = None
    if checkpoint_filepath is not None and os.path.exists(checkpoint_filepath):
        logger.info(f"{'_'*10} checkpoint : {checkpoint_filepath} {'_'*10}")
        model = model.load_from_checkpoint(
            checkpoint_path=checkpoint_filepath, config=pipeline_config
        )
    else:
        logger.info( f"{'_'*10} loading model without checkpoint {'_'*10}")
    
    logger.info("created model")
    
    data_module = CollisionDataModule(train_loader, val_loader, test_loader)
    trainer.fit(
        model=model,
        datamodule = data_module 
    )


#====================================================================================#


if __name__ == "__main__":
    mode = "debug"
    logger.info(f"Running {experiment.name=} in {mode=}")
    run(experiment, mode=mode)