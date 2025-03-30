import os
from tqdm import tqdm
from glob import glob
from loguru import logger
import pandas as pd
import torch
from torch.utils.data import DataLoader
from constants import column_headers
from datasets.veins_dataset import VeinsDataset, VeinsDatasetConfig
device = "cuda"


#====================================================================================#


dataset = VeinsDataset(
    config=VeinsDatasetConfig(
        csv_paths=glob("assets/ScenarioA/*.csv"),
        column_headers=column_headers,
        timesteps=20,
        seed=42,
    )
)
test_dataset = dataset._get_dataset("test")

batch_size = 64
num_workers=32

test_loader = DataLoader(
    test_dataset, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True
)
logger.info(f"created test dataloader, batches: {len(test_loader):_}")


#====================================================================================#


from models.lstm import LSTM, LSTMConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

net = LSTM(
    net_config=LSTMConfig(
        input_dim=9,
        hidden_dim=256, # 64 
        num_layers=12,  # 3
        output_dim=1,
        dropout_prob=0.3
    )
)
net.eval()

exp_name = 'exp0_run2__lstm_12layer_256hiddim'
ckpt_name = 'last'
ckpt = torch.load(f'/disk-2/denish/v2x/experiments/exp0/mcheckpoints/{exp_name}/{ckpt_name}.ckpt')
state_dict = {k.replace('net.', ''):v for k, v in ckpt['state_dict'].items()}
net.load_state_dict(state_dict)
net = net.to(device)
logger.info(f"checkpoint {ckpt_name} loaded from {exp_name}")


#====================================================================================#


@torch.no_grad()
def test_batch(batch):
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    y_hat = net(x)
    predicted = (y_hat > 0.5).float()
    y_true = y.cpu().numpy().flatten().tolist()
    y_pred = predicted.cpu().numpy().flatten().tolist()

    # Calculate test metrics with safe handling
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

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
    average_prediction_time = predicted_collisions
    APT = average_prediction_time / total_collisions if total_collisions else 0
    
    return {
        'test_accuracy': accuracy,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'test_tp': TP,
        'test_fp': FP,
        'test_cpp': CPP,
        'test_cdp': CDP,
        'test_apt': APT
    }


#====================================================================================#


if __name__ == "__main__":
    metrics_avg = {
        'test_accuracy': 0,
        'test_precision': 0,
        'test_recall': 0,
        'test_f1': 0,
        'test_tp': 0,
        'test_fp': 0,
        'test_cpp': 0,
        'test_cdp': 0,
        'test_apt': 0
    }

    ix = 0
    for batch in tqdm(test_loader):  # Start at 1 to avoid division by zero
        metrics = test_batch(batch)
        metrics_avg = {k: metrics_avg[k] + metrics[k] for k in metrics_avg}

    metrics_avg = {k: metrics_avg[k]// len(test_loader) for k in metrics_avg}
    results_dir = f'results/{exp_name}_{ckpt_name}'
    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(metrics_avg, index=[0])  
    df.to_csv(f"{results_dir}/metrics.csv", index=False)
    logger.info("Metrics saved to metrics.csv")
