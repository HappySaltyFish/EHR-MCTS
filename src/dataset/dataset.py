"""
Dataset classes for DelphiSMURF training.

This module provides PyTorch Dataset classes that handle time normalization
during data loading, following the normalization logic of code B as closely
as possible.
"""

import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


EVENT_PAD = 0
TIME_PAD = -10000.0
GAP_PAD = 0.0
GAP_THRESH = -5.0


def get_p2i(data):
    """
    Get patient to index mapping from dataset.

    Args:
        data: numpy array of shape (n_events, 3) with columns [patient_id, age, token]

    Returns:
        numpy array of shape (n_patients, 2) with [start_idx, length]
    """
    if len(data) == 0:
        return np.empty((0, 2), dtype=np.int64)

    p2i = []
    start = 0
    for i in range(1, len(data)):
        if data[i, 0] != data[i - 1, 0]:
            p2i.append([start, i - start])
            start = i
    p2i.append([start, len(data) - start])
    return np.array(p2i, dtype=np.int64)


class DatasetSmurf(Dataset):

    def __init__(
        self,
        data_dir,
        split='train',
        normalize='normal',
        num_samples=None,
        time_mean=None,
        log_mean=None,
        log_std=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.normalize = normalize

        self.time_mean = time_mean
        self.log_mean = log_mean
        self.log_std = log_std

        data_path = os.path.join(data_dir, f'{split}.bin')
        self.data = np.memmap(data_path, dtype=np.uint32, mode='r').reshape(-1, 3)
        self.p2i = get_p2i(self.data)

        if num_samples is not None:
            self.p2i = self.p2i[:num_samples]

        self._compute_time_stats()

        print(
            f"[DatasetSmurf] {split} set: {len(self.p2i)} patients, "
            f"normalize={normalize}, time_mean={self.time_mean:.6f}"
        )

    def _iter_patient_ages(self):
        for idx in range(len(self.p2i)):
            start_idx, length = self.p2i[idx]
            ages = self.data[start_idx:start_idx + length, 1].astype(np.float32)
            ages = np.sort(ages)
            yield ages

    def _compute_time_stats(self):
        time_gaps = []

        for ages in self._iter_patient_ages():
            if len(ages) >= 2:
                gaps = np.diff(ages)
                gaps = gaps[gaps > 0]
                if len(gaps) > 0:
                    time_gaps.extend(gaps.tolist())

        if len(time_gaps) == 0:
            time_gaps = np.array([365.25], dtype=np.float32)
        else:
            time_gaps = np.array(time_gaps, dtype=np.float32)

        if self.time_mean is None:
            self.time_mean = float(time_gaps.mean().item())
        self.time_mean = max(float(self.time_mean), 1e-8)

        if self.normalize == 'normal':
            time_flat = time_gaps / self.time_mean

            self.time_min = float(np.min(time_flat))
            self.time_max = float(np.max(time_flat))
            self.time_mean_norm = float(np.mean(time_flat))
            self.time_std = float(np.std(time_flat))
            self.time_median = float(np.quantile(time_flat, 0.5))
            self.time_quantile_05 = float(np.quantile(time_flat, 0.05))
            self.time_quantile_95 = float(np.quantile(time_flat, 0.95))
            self.time_quantile_99 = float(np.quantile(time_flat, 0.99))
            self.time_quantile = np.array(
                [np.quantile(time_flat, i * 0.1) for i in range(1, 10)],
                dtype=np.float32,
            )

        elif self.normalize == 'log':
            time_flat = time_gaps / self.time_mean
            log_time_flat = np.log(time_flat + 1e-9)

            if self.log_mean is None:
                self.log_mean = float(np.mean(log_time_flat).item())
            if self.log_std is None:
                self.log_std = float(np.std(log_time_flat).item())

            if self.log_std < 1e-8:
                self.log_std = 1.0

            time_flat_z = (log_time_flat - self.log_mean) / self.log_std

            self.time_min = float(np.min(time_flat_z))
            self.time_max = float(np.max(time_flat_z))
            self.time_mean_norm = float(np.mean(time_flat_z))
            self.time_std = float(np.std(time_flat_z))
            self.time_median = float(np.quantile(time_flat_z, 0.5))
            self.time_quantile_05 = float(np.quantile(time_flat_z, 0.05))
            self.time_quantile_95 = float(np.quantile(time_flat_z, 0.95))
            self.time_quantile_99 = float(np.quantile(time_flat_z, 0.99))
            self.time_quantile = np.array(
                [np.quantile(time_flat_z, i * 0.1) for i in range(1, 10)],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Unsupported normalize mode: {self.normalize}")

    def _normalize_abs_time(self, ages):
        if self.normalize == 'normal':
            return ages / self.time_mean
        elif self.normalize == 'log':
            return ages
        return ages

    def _normalize_time_gap(self, time_gap):
        if self.normalize == 'normal':
            return time_gap / self.time_mean
        elif self.normalize == 'log':
            return (np.log(time_gap / self.time_mean + 1e-9) - self.log_mean) / self.log_std
        return time_gap

    def __len__(self):
        return len(self.p2i)

    def __getitem__(self, idx):
        start_idx, length = self.p2i[idx]

        rows = self.data[start_idx:start_idx + length]
        tokens = rows[:, 2].astype(np.int64)
        ages = rows[:, 1].astype(np.float32)

        sort_idx = np.argsort(ages, kind='stable')
        tokens = tokens[sort_idx]
        ages = ages[sort_idx]

        ages_out = self._normalize_abs_time(ages)

        if len(ages) >= 2:
            time_gap = np.diff(ages)
            time_gap = np.clip(time_gap, 0.0, None)
            time_gap = self._normalize_time_gap(time_gap.astype(np.float32))
        else:
            time_gap = np.zeros((0,), dtype=np.float32)

        return (
            torch.tensor(tokens, dtype=torch.long),
            torch.tensor(ages_out, dtype=torch.float32),
            torch.tensor(time_gap, dtype=torch.float32),
        )


def collate_fn_smurf_thp(batch):
    event_types, event_times, time_gaps = zip(*batch)

    max_len = max(x.size(0) for x in event_types)
    max_gap_len = max(max_len - 1, 0)

    bt = []
    bg = []
    by = []

    for y, t, g in zip(event_types, event_times, time_gaps):
        pad_len = max_len - y.size(0)
        gap_pad_len = max_gap_len - g.size(0)

        y = torch.cat([y, torch.full((pad_len,), EVENT_PAD, dtype=torch.long)])
        t = torch.cat([t, torch.full((pad_len,), TIME_PAD, dtype=torch.float32)])
        g = torch.cat([g, torch.full((gap_pad_len,), GAP_PAD, dtype=torch.float32)])

        by.append(y)
        bt.append(t)
        bg.append(g)

    return torch.stack(by), torch.stack(bt), torch.stack(bg)


def get_dataloader_smurf(data_dir, batch_size=128, split='train',
                         no_event_token_rate=5, normalize='normal', shuffle=True, num_workers=0,
                         time_mean=None):
    dataset = DatasetSmurf(
        data_dir=data_dir,
        split=split,
        normalize=normalize,
        time_mean=time_mean
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn_smurf_thp
    )

    return loader

