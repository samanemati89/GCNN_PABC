import os
import dgl
import torch
import numpy as np
from dgl.data import DGLDataset


class Polar(DGLDataset):

	def __init__(self, root, split, num_nodes=189):
		# super(Polar, self).__init__()
		self.root = root
		self.split = split
		self.num_nodes = num_nodes
		self.graphs_base = os.path.join(self.root, "graphs", self.split)
		self.node_features_base = os.path.join(self.root, "node_features", self.split)
		

		self.items_list, self.label_names = self.get_matirx_list()

	def	get_matirx_list(self):
		
		items_list = []
		label_names = []

		counter = 0
		for root, dirs, files in os.walk(self.graphs_base, topdown=True):
			nftrs_root = os.path.join(self.node_features_base, os.path.split(root)[1])
			
			if counter == 0:
				label_names = dirs

			for file in files:
				if file.endswith(".csv"):
					items_list.append(
						{
							"graph": os.path.join(root, file),
							"node_features": os.path.join(nftrs_root, file),
							"label": counter - 1,
							"label_name": label_names[counter - 1]	
						}
					)
			counter += 1

		return items_list, label_names

	def __getitem__(self, index):

		graph_dir = self.items_list[index]["graph"]
		nftrs_dir = self.items_list[index]["node_features"]
		label = self.items_list[index]["label"]
		label_name = self.items_list[index]["label_name"]

		node_features = np.loadtxt(nftrs_dir, delimiter=",")

		edge_weights = np.loadtxt(graph_dir, delimiter=",")
		edge_weights = np.squeeze(edge_weights.reshape(1, -1))

		src = [[0 for i in range(self.num_nodes)] for j in range(self.num_nodes)]

		for i in range(len(src)):
			for j in range(len(src[i])):
				src[i][j] = i
		src = np.array(src).flatten()

		det = [[i for i in range(self.num_nodes)] for j in range(self.num_nodes)]
		det = np.array(det).flatten()

		u, v = (torch.tensor(src), torch.tensor(det))
		g = dgl.graph((u, v))

        # add node features and edge features
		g.ndata["node_features"] = torch.from_numpy(node_features)
		g.edata["edge_weights"] = torch.from_numpy(edge_weights)

		return g, label, label_name

	def __len__(self):
		return len(self.items_list)

def main():
	dataset = Polar("./dataset", "train")
	print(f"Size of train set: {len(dataset):d}")
	dataiter = iter(dataset)
	g, label, label_name = next(dataiter)
	print(g, label, label_name)


if __name__ == "__main__":
	main()
