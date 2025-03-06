import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GCN, GCNV2, GIN
from dataloader import Polar
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import classification_report
from saliency import compute_saliency_maps

# Processor configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading datasets
dataset = Polar("/home/serfani/Documents/polar/dataset", "test")
dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=True, drop_last=False)

# X, y = next(iter(dataloader))

# Instantiating Model
# model = GCN(427, 256)
# model = GCNV2(427, 256)
model = GIN(427, 128, 2)

# snapshots/10-11-2023/
with open('snapshots/GIN/checkpoints.pth.tar', mode='rb') as f:

    checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print('Learning Rate: ', checkpoint['learning_rate'])
    print('Number of Epochs: ', checkpoint['epoch'])

patient_cases = [] # label -> 0
healthy_cases = [] # label -> 1

patient_cases_preds = []
healthy_cases_preds = []

for X, y in dataloader:
    
    saliency_map, pred = compute_saliency_maps(X, y, model)
    
    if y == 0:
        
        if pred == 0:
            patient_cases_preds.append(1)
        
        else:
            patient_cases_preds.append(0)
        
        patient_cases.append(saliency_map.detach().cpu().numpy())
    
    else:
        if pred == 1:
            healthy_cases_preds.append(1)
        
        else:
            healthy_cases_preds.append(0)

        healthy_cases.append(saliency_map.detach().cpu().numpy())

print(patient_cases_preds)
print(healthy_cases_preds)

patient_cases = np.array(patient_cases) * 1.0E7
healthy_cases = np.array(healthy_cases) * 1.0E7

patient_cases_preds = np.array(patient_cases_preds)
healthy_cases_preds = np.array(healthy_cases_preds)

patient_saliency_data = np.concatenate((patient_cases, patient_cases_preds.reshape(-1, 1)), axis=1)
healthy_saliency_data = np.concatenate((healthy_cases, healthy_cases_preds.reshape(-1, 1)), axis=1)
print(patient_saliency_data.shape, healthy_saliency_data.shape)

np.savetxt("patient_saliency_maps.csv", patient_saliency_data, delimiter=",")
np.savetxt("healthy_saliency_maps.csv", healthy_saliency_data, delimiter=",")

exit()
healthy_cases_mean = np.array(healthy_cases).mean(axis=0)
patient_cases_mean = np.array(patient_cases).mean(axis=0)



print(healthy_cases.min(), healthy_cases.max())
print(patient_cases.min(), patient_cases.max())

# 2.1304858 4.671248
# 3.9710696 19.167086


print(patient_cases_mean.shape)
fig, ax1 = plt.subplots()

ax1.plot(patient_cases[0], 'r.', label='patient')

ax2 = ax1.twinx()
ax2.plot(healthy_cases[5], 'g.', label='healthy')

# plt.grid(True)
# plt.legend()
plt.show()

exit()


import copy
import matplotlib.colors as mcolors

healthy_cases_mean = np.array(healthy_cases).mean(axis=0, keepdims=True)
patient_cases_mean = np.array(patient_cases).mean(axis=0, keepdims=True)

data = np.concatenate([healthy_cases_mean, patient_cases_mean], axis=0)

num_vars = patient_cases_mean.shape[1]
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

fig, ax = plt.subplots(figsize=(15, 15), subplot_kw=dict(polar=True))

for idx, arr in enumerate(data):
    print(arr.shape)
    values = np.append(arr, arr[0])
    angles_adjusted = np.append(angles, angles[0])
    
    if idx == 0:
        
        ax.scatter(angles_adjusted, values*1.0E7, linewidth=0.5, label='Healthy Cases', alpha=1, marker='.')
    
    else:
        pass
        # ax.plot(angles_adjusted, values, linewidth=1, label='Patient Cases', alpha=0.4, linestyle='--', color='orange')

# # Set the angle labels
# ax.set_ylim(data.min(), data.max())
# ax.set_yticks([data[0].min(), data[0].max()])
ax.set_xticks(angles)
ax.set_xticklabels([i for i in range(189)], fontsize=6)  # Use column names for angle labels
# ax.xaxis.labelpad = 5000 

plt.show()

# ax.tick_params(axis='both', labelsize=14, rotation = 0, pad = 25)
# ax.set_rlabel_position(90)

# legend = ax.legend(loc= (1.3,-0.085), fontsize = 12,title='Cutoff Value')
# plt.setp(legend.get_title(), fontsize=14)
# plt.savefig('Outputs\FirstStage_Cutoff_Optimization.jpeg', bbox_inches = 'tight')
# #%%