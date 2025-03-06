import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GCN, GCNV2, GIN
from dataloader import Polar
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report

# Processor configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading datasets
train_dataset = Polar("/home/serfani/Documents/polar/dataset", "train")
test_dataset = Polar("/home/serfani/Documents/polar/dataset", "test")

# Creating dataloaders
train_dataloader = GraphDataLoader(train_dataset, batch_size=20, shuffle=True, drop_last=False)
test_dataloader = GraphDataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

# Instantiating Model
# model = GCN(427, 256).to(device)
# model = GCNV2(427, 256).to(device)
model = GIN(427, 128, 2).to(device)

# snapshots/10-13-2023/
with open('checkpoints.pth.tar', mode='rb') as f:

    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('Learning Rate: ', checkpoint['learning_rate'])
    print('Number of Epochs: ', checkpoint['epoch'])


with open('logger.npy', mode='rb') as f:
    loss = np.load(f)
    acc = np.load(f)


fig, ax1 = plt.subplots()

lns1 = ax1.plot(loss, 'r', label='loss')

ax2 = ax1.twinx()
lns2 = ax2.plot(acc, 'g', label='acc')

ax1.set_ylabel('Loss')
ax1.yaxis.label.set_color('red')
ax1.tick_params(axis='y', colors='red')

ax2.set_ylabel('Accuracy')
ax2.yaxis.label.set_color('green')
ax2.tick_params(axis='y', colors='green')

lns = lns1 + lns2
labs = [l.get_label() for l in lns]

ax1.legend(lns, labs, loc=1)
plt.savefig('logger.svg', format='svg', bbox_inches='tight', pad_inches=0.1)
plt.show()


model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    n_correct = 0
    n_samples = 0
    for graphs, labels in test_dataloader:
        graphs = graphs.to(device)
        labels = labels.to(device)
        # outputs = model(graphs)
        outputs = model(graphs, graphs.ndata["feat"])

        _, predicted = torch.max(outputs, 1)
        n_samples += len(labels)
        n_correct += (predicted == labels).sum().item()

        y_true.append(labels.item())
        y_pred.append(predicted.item())

    
acc = n_correct / n_samples
print(f'Accuracy of the network: {acc:.2%}')

final_report = classification_report(y_true, y_pred, target_names= train_dataset.label_names)
print(final_report)