import torch
import pytest
from src.data.make_dataset import CorruptMnist
from torch.utils.data import Dataset
import os 
from tests import _PATH_DATA

@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_len():

    # read data files from path
    train_set = torch.load("data/processed/train_set.pt")
    test_set = torch.load("data/processed/test_set.pt")

    assert len(train_set) == 25000, "Train set doesn't have the expected length"
    assert len(test_set) == 5000, "Test set doesn't have the expected length"