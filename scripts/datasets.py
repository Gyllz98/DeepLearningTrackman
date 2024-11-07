from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from torch.utils.data import Dataset

from data_management import get_observation_nums

class SpectrogramDataset(Dataset):
    def __init__(self,
                 transform,
                 stmf_data_path: Path,
                 data_dir: Path = None,
                 observation_nums: Iterable = None,
                 csv_delimiter: str = ",") -> None:
        
        if data_dir is None and observation_nums is None:
            raise ValueError("Either `data_dir` or `observation_nums` mus be different from None")
        
        if data_dir is not None:
            observation_nums = get_observation_nums(data_dir)
        self.observation_nums = observation_nums

        self.stmf_data  = pd.read_csv(stmf_data_path, delimiter=csv_delimiter).iloc[observation_nums]
        self.targets = self.stmf_data.BallVr.to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> dict:
        spectrogram, target = self._get_item_helper(idx)
        sample = {"spectrogram": spectrogram, "target": target}

        return sample
    
    def _get_item_helper(self, idx: int) -> tuple:
        stmf_row = self.stmf_data.iloc[idx]

        spectrogram = self.transform(stmf_row)
        target = self.targets[idx]
        target = torch.tensor(target, dtype=torch.float32)

        return spectrogram, target
