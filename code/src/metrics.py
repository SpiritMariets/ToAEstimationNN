import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss, total_mae, total_valid_mpcs = 0.0, 0.0, 0

    for batch in dataloader:
        cir = batch['cir'].to(device, non_blocking=True)
        delays = batch['delays'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        mask = mask[:, 1:]
        target = delays[:, 1:]
        delays = delays[:, :-1]

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            predictions = model(cir, delays)
            raw_loss = loss_fn(predictions, target)
            loss = (raw_loss * mask).sum() / mask.sum().clamp(min=1) if raw_loss.dim() > 0 else raw_loss

        mae = (torch.abs(predictions - target) * mask).sum().item()
        valid_count = mask.sum().item()

        total_loss += loss.item() * valid_count
        total_mae += mae
        total_valid_mpcs += valid_count

    return {
        'Loss': total_loss / max(total_valid_mpcs, 1),
        'MAE': total_mae / max(total_valid_mpcs, 1)
    }


def compute_delay_metrics(
    pred_delays: np.ndarray, 
    gt_delays: np.ndarray, 
    miss_penalty: float = 1.0
) -> Dict[str, float]:
    m_pred = len(pred_delays)
    m_gt = len(gt_delays)

    if m_pred == 0 and m_gt == 0:
        return {'mae_matched': 0.0, 'abs_count_err': 0, 'otd': 0.0}
    
    if m_pred == 0 or m_gt == 0:
        return {
            'mae_matched': miss_penalty,
            'abs_count_err': abs(m_pred - m_gt),
            'otd': miss_penalty * max(m_pred, m_gt)
        }

    cost_matrix = np.abs(pred_delays[:, None] - gt_delays[None, :])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matched_mae = cost_matrix[row_ind, col_ind].mean()
    count_err = abs(m_pred - m_gt)
    
    total_match_cost = cost_matrix[row_ind, col_ind].sum()
    unmatched_penalty = count_err * miss_penalty
    otd_score = (total_match_cost + unmatched_penalty) / max(m_gt, 1)

    return {
        'mae_matched': float(matched_mae),
        'abs_count_err': int(count_err),
        'otd': float(otd_score)
    }


@torch.no_grad()
def evaluate_autoregressive(
    model: torch.nn.Module,
    dataloader: DataLoader,
    max_len: int = 50,
    threshold: float = 0.95,
    scale_factor: float = 1.0
) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    total_matched_mae, total_count_err, total_otd, total_samples = 0.0, 0.0, 0.0, 0

    for batch in tqdm(dataloader, desc="Autoregressive Testing"):
        cirs = batch['cir'].to(device)
        gt_delays_batch = batch['delays']
        masks_batch = batch['mask']

        batch_size = cirs.size(0)

        for i in range(batch_size):
            single_cir = cirs[i:i+1]
            gt_mask = masks_batch[i]
            gt_d = gt_delays_batch[i][gt_mask]
            gt_d = gt_d[(gt_d > 0.0) & (gt_d < 1.0)].cpu().numpy()

            pred_d = model.predict(
                src=single_cir,
                max_len=max_len,
                threshold=threshold,
                scale_factor=scale_factor
            ).cpu().numpy()

            metrics = compute_delay_metrics(pred_d, gt_d)

            total_matched_mae += metrics['mae_matched']
            total_count_err += metrics['abs_count_err']
            total_otd += metrics['otd']
            total_samples += 1

    return {
        'Avg_Matched_MAE': total_matched_mae / total_samples,
        'Avg_MPC_Count_Error': total_count_err / total_samples,
        'Avg_OTD_Score': total_otd / total_samples
    }
