import torch 
from model import SimpleCNN
import click
from src.data.make_dataset import CorruptMnist


@click.command()
@click.argument("model_path")
def test(model_path):
        
        test_set = torch.load("data/processed/test_set.pt")
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



# def validation(epoch):
#     model.eval()
#     fin_targets=[]
#     fin_outputs=[]
#     with torch.no_grad():
#         for _, data in enumerate(testing_loader, 0):
#             ids = data['ids'].to(device, dtype = torch.long)
#             mask = data['mask'].to(device, dtype = torch.long)
#             token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#             targets = data['targets'].to(device, dtype = torch.float)
#             outputs = model(ids, mask, token_type_ids)
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
#     return fin_outputs, fin_targets



# for epoch in range(EPOCHS):

#     model.eval()
#     fin_targets=[]
#     fin_outputs=[]
#     with torch.no_grad():
#         for _, data in enumerate(testing_loader, 0):
#             ids = data['ids'].to(device, dtype = torch.long)
#             mask = data['mask'].to(device, dtype = torch.long)
#             token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
#             targets = data['targets'].to(device, dtype = torch.float)
#             outputs = model(ids, mask, token_type_ids)
#             fin_targets.extend(targets.cpu().detach().numpy().tolist())
#             fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())    
#     outputs = np.array(fin_outputs) >= 0.5
#     accuracy = metrics.accuracy_score(targets, outputs)
#     f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
#     f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
#     print(f"Accuracy Score = {accuracy}")
#     print(f"F1 Score (Micro) = {f1_score_micro}")
#     print(f"F1 Score (Macro) = {f1_score_macro}")