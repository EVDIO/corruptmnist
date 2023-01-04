import torch
from model import SimpleCNN
#from data.make_dataset import CorruptMnist
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset
class CorruptMnist(Dataset):
    def __init__(self, train, input_filepath = None, output_filepath = None):

        if train:
            content = []
            for i in range(5):
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

def train():

    # use CUDA if available
    cuda_availability = torch.cuda.is_available()
    if cuda_availability:
        device = torch.device("cuda:{}".format(torch.cuda.current_device()))
    else:
        device = "cpu"

    # read data files from path
    train_set = torch.load("data/processed/train_set.pt")
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=128)
    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    n_epoch = 5
    for epoch in range(n_epoch):
        loss_tracker = []
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            loss_tracker.append(loss.item())
        print(f"Epoch {epoch+1}/{n_epoch}. Loss: {loss}")        
    torch.save(model.state_dict(), 'trained_model.pt')
 
    return model

if __name__ == "__main__":
    train()
    