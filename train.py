import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GCN, GCNV2, GIN, initialize_weights
from dataloader import Polar
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report

# Processor configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading datasets
train_dataset = Polar("/home/serfani/Documents/polar/dataset", "train")
test_dataset = Polar("/home/serfani/Documents/polar/dataset", "test")

# Creating dataloaders
train_dataloader = GraphDataLoader(train_dataset, batch_size=10, shuffle=True, drop_last=False)
test_dataloader = GraphDataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

# Instantiating Model
model = GCN(427, 256).to(device)
model = GCNV2(427, 256).to(device)
model = GIN(427, 128, 2).to(device)


initialize_weights(model)
model.train()


learning_rate = 2.0E-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1.0E-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

loss_fn = torch.nn.CrossEntropyLoss()

logger = {'loss': list(), 'accuracy': list()}

num_epochs = 100
for epoch in range(num_epochs):
    
    n_samples = 0; n_correct = 0
    batch_loss = 0.0
    
    for idx, (graphs, labels) in enumerate(train_dataloader):

        graphs = graphs.to(device)
        labels = labels.to(device)

        # Forward pass
        # outputs = model(graphs)
        outputs = model(graphs, graphs.ndata["feat"])

        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, dim=1)

        n_samples += len(labels)
        n_correct += (predicted == labels).sum().item()

        batch_loss += loss.item()

    scheduler.step()

    # reports:
    acc = n_correct / n_samples
    logger['accuracy'].append(acc)
    logger['loss'].append(batch_loss / len(train_dataloader))

    print (f'Epoch [{epoch+1}/{num_epochs}], ACC: {acc:.2f}, LOSS: {logger["loss"][-1]:.4f}')

with open('logger.npy', mode='wb') as f:
    np.save(f, np.array(logger['loss']))
    np.save(f, np.array(logger['accuracy']))


checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'learning_rate': learning_rate,
    'epoch': num_epochs
}

with open('checkpoints.pth.tar', mode='wb') as f:
    torch.save(checkpoint, f)


fig, ax1 = plt.subplots()

lns1 = ax1.plot(logger['loss'], 'r', label='loss')

ax2 = ax1.twinx()
lns2 = ax2.plot(logger['accuracy'], 'g', label='acc')

lns = lns1 + lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=1)

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