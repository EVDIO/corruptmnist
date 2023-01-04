import torch 
from model import SimpleCNN
import click


@click.command()
@click.argument("model_path")
def test(model_path):
        
        test_set = torch.load("data/processed/test_set.pth")
        dataloader = torch.utils.data.DataLoader(test_set, batch_size=64)
        
        
        # TODO: Implement evaluation logic here
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