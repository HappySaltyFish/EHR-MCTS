import os
import math
import pickle
import numpy as np
import torch
import torch.utils.data

EVENT_PAD = 0
TIME_PAD = -10000.0
GAP_PAD = 0.0


class NHPTHPDataset(torch.utils.data.Dataset):
    """
    NHP dataset in THP style, with normalization logic aligned to code B.
    """

    def __init__(
        self,
        data,
        split="train",
        normalize="normal",
        time_mean=None,
        log_mean=None,
        log_std=None,
    ):
        self.normalize = normalize

        self.time = [
            [float(elem["time_since_start"]) for elem in inst]
            for inst in data
        ]
        self.time_gap = [
            [float(elem["time_since_last_event"]) for elem in inst[1:]]
            for inst in data
        ]
        self.event_type = [
            [int(elem["type_event"]) + 1 for elem in inst]
            for inst in data
        ]

        self.length = len(data)
        self.max_len = max((len(x) for x in self.event_type), default=0)

        time_flat = []
        for gaps in self.time_gap:
            time_flat += gaps
        time_flat = np.asarray(time_flat, dtype=np.float32)

        if time_flat.size == 0:
            time_flat = np.asarray([1.0], dtype=np.float32)

        if split == "train":
            if self.normalize == "normal":
                mean_data = float(time_flat.mean().item()) if time_mean is None else float(time_mean)
                mean_data = max(mean_data, 1e-8)
                self.time_mean = mean_data

                time_flat = time_flat / mean_data
                self.time_gap = [[elem / mean_data for elem in inst] for inst in self.time_gap]
                self.time = [[elem / mean_data for elem in inst] for inst in self.time]

            elif self.normalize == "log":
                mean_data = float(time_flat.mean().item()) if time_mean is None else float(time_mean)
                mean_data = max(mean_data, 1e-8)
                self.time_mean = mean_data

                time_flat = time_flat / mean_data
                log_time_flat = np.log(time_flat + 1e-9)

                mean_log_data = float(log_time_flat.mean().item()) if log_mean is None else float(log_mean)
                std_log_data = float(log_time_flat.std().item()) if log_std is None else float(log_std)
                if std_log_data < 1e-8:
                    std_log_data = 1.0

                self.log_mean = mean_log_data
                self.log_std = std_log_data

                time_flat = (log_time_flat - mean_log_data) / std_log_data
                self.time_gap = [
                    [
                        (math.log(elem / self.time_mean + 1e-9) - self.log_mean) / self.log_std
                        for elem in inst
                    ]
                    for inst in self.time_gap
                ]
                # code B: log mode does not normalize absolute time

            else:
                raise ValueError(f"Unsupported normalize mode: {self.normalize}")

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

        else:
            if time_mean is None:
                raise ValueError("time_mean must be provided for non-train splits")
            self.time_mean = max(float(time_mean), 1e-8)

            if self.normalize == "normal":
                self.time_gap = [[elem / self.time_mean for elem in inst] for inst in self.time_gap]
                self.time = [[elem / self.time_mean for elem in inst] for inst in self.time]

                time_flat = time_flat / self.time_mean
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

            elif self.normalize == "log":
                if log_mean is None or log_std is None:
                    raise ValueError("log_mean and log_std must be provided for log mode on non-train splits")

                self.log_mean = float(log_mean)
                self.log_std = float(log_std)
                if self.log_std < 1e-8:
                    self.log_std = 1.0

                self.time_gap = [
                    [
                        (math.log(elem / self.time_mean + 1e-9) - self.log_mean) / self.log_std
                        for elem in inst
                    ]
                    for inst in self.time_gap
                ]

                # self.time = [
                #     [
                #         (math.log(elem / self.time_mean + 1e-9) - self.log_mean) / self.log_std
                #         for elem in inst
                #     ]
                #     for inst in self.time
                # ]

                time_flat = (np.log(time_flat / self.time_mean + 1e-9) - self.log_mean) / self.log_std
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

            else:
                raise ValueError(f"Unsupported normalize mode: {self.normalize}")

        print(
            f"[NHPTHPDataset] {split}: {self.length} sequences, "
            f"max_len={self.max_len}, normalize={self.normalize}, "
            f"time_mean={self.time_mean:.6f}"
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        event_time = np.asarray(self.time[idx], dtype=np.float32)
        time_gap = np.asarray(self.time_gap[idx], dtype=np.float32)
        event_type = np.asarray(self.event_type[idx], dtype=np.int64)

        return (
            torch.tensor(event_type, dtype=torch.long),
            torch.tensor(event_time, dtype=torch.float32),
            torch.tensor(time_gap, dtype=torch.float32),
        )


def _pad_float_sequences(seqs, pad_value):
    max_len = max((len(x) for x in seqs), default=0)
    out = np.full((len(seqs), max_len), pad_value, dtype=np.float32)
    for i, seq in enumerate(seqs):
        out[i, :len(seq)] = seq
    return torch.tensor(out, dtype=torch.float32)


def _pad_long_sequences(seqs, pad_value=0):
    max_len = max((len(x) for x in seqs), default=0)
    out = np.full((len(seqs), max_len), pad_value, dtype=np.int64)
    for i, seq in enumerate(seqs):
        out[i, :len(seq)] = seq
    return torch.tensor(out, dtype=torch.long)


def collate_fn_nhp_thp(batch):
    event_type, event_time, time_gap = zip(*batch)
    event_type = _pad_long_sequences(event_type, pad_value=EVENT_PAD)
    event_time = _pad_float_sequences(event_time, pad_value=TIME_PAD)
    time_gap = _pad_float_sequences(time_gap, pad_value=GAP_PAD)
    return event_type, event_time, time_gap


def get_dataloader_nhp_thp(
    data_dir,
    batch_size=16,
    split="train",
    normalize="normal",
    shuffle=True,
    num_workers=0,
    time_mean=None,
    log_mean=None,
    log_std=None
):
    split_map = {
        "train": ("train.pkl", "train"),
        "dev":   ("dev.pkl", "dev"),
        "val":   ("dev.pkl", "dev"),
        "test":  ("test.pkl", "test"),
    }
    if split not in split_map:
        raise ValueError(f"Unsupported split: {split}")

    fname, dict_key = split_map[split]
    pkl_path = os.path.join(data_dir, fname)

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin-1")

    num_types = int(raw["dim_process"])
    data = raw[dict_key]

    dataset = NHPTHPDataset(
        data=data,
        split=split,
        normalize=normalize,
        time_mean=time_mean,
        log_mean=log_mean,
        log_std=log_std
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_nhp_thp,
        pin_memory=True,
    )

    return loader, num_types
