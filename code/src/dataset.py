import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, Dict, Any
import scipy.io as sio
from qdchannel import QdChannel  # Ваша внешняя библиотека

SIGNAL_POWER = 10
SPEED_OF_LIGHT = 299_792_458

def set_global_seed(seed: int) -> None:
    """Установка глобального seed для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ChannelsDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        bandwidth: float,
        carriers: int, 
        noise: Optional[Tuple[float, float]] = None
    ):
        self.bandwidth = bandwidth
        self.carriers = carriers
        self.max_delay = carriers / bandwidth
        self.noise = noise
        self.channels_paths = sorted(glob.glob(dataset_path))

    def __len__(self) -> int:
        return len(self.channels_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.channels_paths[idx]
        channel = QdChannel.from_mat(path).item()

        if self.noise is not None:
            noise = np.random.uniform(self.noise[0], self.noise[1])
        else:
            noise = self.noise

        cfr = channel.fr(self.bandwidth, self.carriers).squeeze()
        power = np.mean(np.abs(cfr) ** 2)
        cfr = np.sqrt(SIGNAL_POWER) * cfr / np.sqrt(power)

        if noise is not None:
            noise_var = SIGNAL_POWER / 10 ** (noise / 10)
            noise_signal = np.sqrt(noise_var / 2) * (np.random.randn(self.carriers) + 1j * np.random.randn(self.carriers))
            cfr += noise_signal

        cir = np.fft.ifft(cfr)
        cir = torch.from_numpy(np.stack([cir.real, cir.imag], axis=0)).float()

        delays = channel.delays.squeeze() / self.max_delay
        indices = np.argsort(delays)

        delays = torch.from_numpy(delays[indices]).float()
        coeffs = channel.coeffs.squeeze()[indices]
        coeffs = torch.from_numpy(np.stack([coeffs.real, coeffs.imag], axis=0)).float()

        return {'cir': cir, 'delays': delays, 'coeffs': coeffs}


class CIRCollate:
    def __init__(self, pad_val: float = 1.0):
        self.pad_val = pad_val

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        cirs, delays_list, lengths = [], [], []

        for item in batch:
            cir = item['cir']
            cirs.append(cir)

            d = item['delays']
            sos = torch.tensor([0.0], dtype=d.dtype, device=d.device)
            eos = torch.tensor([1.0], dtype=d.dtype, device=d.device)
            d = torch.cat([sos, d, eos])

            delays_list.append(d)
            lengths.append(len(d))

        batch_cir = torch.stack(cirs, dim=0)
        batch_delays = pad_sequence(delays_list, batch_first=True, padding_value=self.pad_val)
        batch_lengths = torch.tensor(lengths, dtype=torch.long)

        max_len = batch_delays.size(1)
        batch_size = len(batch)
        mask = torch.arange(max_len).expand(batch_size, max_len) < batch_lengths.unsqueeze(1)

        return {
            'cir': batch_cir,
            'delays': batch_delays,
            'lengths': batch_lengths,
            'mask': mask
        }
