import os

from src.models.model import CorruptMnist
import pytest
import torch
from torch.utils.data import Dataset

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_len():

    # read data files from path
    train_set = torch.load("data/processed/train_set.pt")
    test_set = torch.load("data/processed/test_set.pt")

    assert len(train_set) == 25000, "Train set doesn't have the expected length"
    assert len(test_set) == 5000, "Test set doesn't have the expected length"


class CorruptMnist(Dataset):
    def __init__(self, train, input_filepath = None, output_filepath = None):

        if train:
            content = []
            for i in range(8):
                content.append(np.load(os.path.join(input_filepath, 'train_{}.npz'.format(i)), allow_pickle=True))
            data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
            targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
        else:
            content = np.load(os.path.join(input_filepath, 'test.npz'), allow_pickle=True)
            data = torch.tensor(content['images']).reshape(-1, 1, 28, 28)
            targets = torch.tensor(content['labels'])
            
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return self.targets.numel()
    
    def __getitem__(self, idx):

        return self.data[idx].float(), self.targets[idx]
        