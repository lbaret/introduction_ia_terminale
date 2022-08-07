from typing import Tuple

import torch
from torch.utils.data import Dataset


class FakeNewsDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, targets: torch.Tensor, **torch_dataset) -> None:
        super(FakeNewsDataset, self).__init__(**torch_dataset)
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.attention_mask[idx], self.targets[idx]

    def __len__(self) -> int:
        return len(self.input_ids)
