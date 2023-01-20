import torch
from model import SimpleCNN
from src.data.make_dataset import CorruptMnist
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import torch
import os
from torchvision import transforms
from torch.utils.data import Dataset


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
    torch.save(model.state_dict(), 'models/trained_model.pt')
 
    return model

if __name__ == "__main__":
    train()
