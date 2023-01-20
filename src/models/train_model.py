import torch
import yaml

from src.data.make_dataset import CorruptMnist
from src.models.model import SimpleCNN

# import wandb


def train(train_set_path: str, model_path: str, config_path: str,
          batch_size: int = 128, n_epoch: int = 5, lr: float = 1e-4):
    """
    Function to train the model on the given dataset
    and save the trained model.

    Parameters:
        train_set_path (str): path of the train set file
        model_path (str): path to save the trained model
        batch_size (int): batch size for training (default: 128)
        n_epoch (int): number of training epochs (default: 5)
        lr (float): learning rate for the optimizer (default: 1e-4)

    Returns:
        model: trained model
    """
    # Load the configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract the parameters for the CNN
    input_channels = config["input_channels"]
    output_channels = config["output_channels"]
    kernel_size = config["kernel_size"]
    n_hidden = config["n_hidden"]
    dropout = config["dropout"]

    # read data files from path
    train_set = torch.load(train_set_path)
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    model = SimpleCNN(input_channels, output_channels, 
                      kernel_size, n_hidden, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

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
    torch.save(model.state_dict(), model_path)

    return model


if __name__ == "__main__":
     # Load the configuration file
    with open("conf/config_train.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # set the train pat and the model path
    train_set_path = config['train_set_path']
    model_path = config['model_path']
    batch_size = config['batch_size']
    n_epoch = config['n_epoch']
    lr = config['lr']

    config_path = "conf/config_model.yaml"
    train(train_set_path, model_path, config_path, batch_size, n_epoch, lr)

