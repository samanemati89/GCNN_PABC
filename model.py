import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn import GraphConv
from dgl.nn.pytorch.glob import SumPooling


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (GraphConv)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes=2):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.bn1 = nn.BatchNorm1d(h_feats)
        self.conv2 = GraphConv(h_feats, int(h_feats / 2))
        self.bn2 = nn.BatchNorm1d(int(h_feats / 2))
        self.conv3 = GraphConv(int(h_feats / 2), int(h_feats / 4))
        self.bn3 = nn.BatchNorm1d(int(h_feats / 4))
        self.conv4 = GraphConv(int(h_feats / 4), int(h_feats / 8))
        self.bn4 = nn.BatchNorm1d(int(h_feats / 8))
        self.conv5 = GraphConv(int(h_feats / 8), int(h_feats / 16))
        self.bn5 = nn.BatchNorm1d(int(h_feats / 16))
        self.conv6 = GraphConv(int(h_feats / 16), int(h_feats / 32))
        self.bn6 = nn.BatchNorm1d(int(h_feats / 32))
        self.relu = nn.PReLU()
        self.classify = nn.Linear(int(h_feats / 32), num_classes)

    def forward(self, g):
        # Apply graph convolution and activation.
        h = self.conv1(g, g.ndata["feat"])
        h = self.bn1(h)
        h = self.relu(h)

        h = self.conv2(g, h)
        h = self.bn2(h)
        h = self.relu(h)

        h = self.conv3(g, h)
        h = self.bn3(h)
        h = self.relu(h)

        h = self.conv4(g, h)
        h = self.bn4(h)
        h = self.relu(h)

        h = self.conv5(g, h)
        h = self.bn5(h)
        h = self.relu(h)

        h = self.conv6(g, h)
        h = self.bn6(h)
        h = self.relu(h)

        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg) 


class GCNV2(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes=2, dropout=0.5, weight_decay=5e-4):
        super(GCNV2, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.bn1 = nn.BatchNorm1d(h_feats)
        self.conv2 = GraphConv(h_feats, int(h_feats / 2))
        self.bn2 = nn.BatchNorm1d(int(h_feats / 2))
        self.conv3 = GraphConv(int(h_feats / 2), int(h_feats / 4))
        self.bn3 = nn.BatchNorm1d(int(h_feats / 4))
        self.relu = nn.LeakyReLU(0.2)
        self.classify = nn.Linear(int(h_feats / 4), num_classes)
        self.dropout = nn.Dropout(dropout)
        self.weight_decay = weight_decay

    def forward(self, g):
        # Apply graph convolution and activation.
        h = self.conv1(g, g.ndata["feat"])
        h = self.bn1(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.conv2(g, h)
        h = self.bn2(h)
        h = self.relu(h)
        h = self.dropout(h)

        h = self.conv3(g, h)
        h = self.bn3(h)
        h = self.relu(h)
        h = self.dropout(h)

        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            logits = self.classify(hg)
            return logits


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(num_layers - 1):  # excluding the input layer
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                self.linear_prediction.append(nn.Linear(input_dim, output_dim))
            else:
                self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))
        return score_over_layer