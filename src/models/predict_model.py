import click
import torch

from src.data.make_dataset import CorruptMnist
from src.models.model import SimpleCNN


@click.command()
@click.argument("model_path", type=click.Path(exists=True, dir_okay=False,
                file_okay=True, readable=True, resolve_path=True))
def test(model_path: str):
    """
    Function to test the model on the test set and print the accuracy.

    Parameters:
        model_path (str): path of the trained model file
    """
    test_set = torch.load("data/processed/test_set.pt")
    dataloader = torch.utils.data.DataLoader(test_set, batch_size=64)

    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))

    correct, total = 0, 0
    for batch in dataloader:
        x, y = batch
        preds = model(x)
        preds = preds.argmax(dim=-1)

        correct += (preds == y).sum().item()
        total += y.numel()

    print(f"Test set accuracy {correct/total}")


if __name__ == "__main__":
    test()
