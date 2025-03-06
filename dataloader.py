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
		
		self.gclasses = len(self.label_names)

	def	get_matirx_list(self):
		
		items_list = []
		label_names = []

		for idx, (root, dirs, files) in enumerate(os.walk(self.graphs_base, topdown=True)):
			nftrs_root = os.path.join(self.node_features_base, os.path.split(root)[1])
			
			if idx == 0:
				label_names = dirs

			for file in files:
				if file.endswith(".csv"):
					items_list.append(
						{
							"graph": os.path.join(root, file),
							"node_features": os.path.join(nftrs_root, file),
							"label": idx - 1,
							"label_name": label_names[idx - 1]	
						}
					)

		return items_list, label_names

	def __getitem__(self, index):

		graph_dir = self.items_list[index]["graph"]
		nftrs_dir = self.items_list[index]["node_features"]
		label = self.items_list[index]["label"]
		label_name = self.items_list[index]["label_name"]

		node_features = np.loadtxt(nftrs_dir, delimiter=",", dtype=np.float32)
		edge_weights = np.loadtxt(graph_dir, delimiter=",", dtype=np.float32)
		
		# node_features = (node_features - 8512.963) / 2559.9653 
		# edge_weights = (edge_weights - 2.862617) / 9.373531 

		# MinMax Normalization
		node_features = 2 * ((node_features - 0.0) / 20608.0) -1 
		edge_weights = 2 * ((edge_weights - 0.0) / 503.96) -1 		

		edge_weights = np.ravel(edge_weights)

		src = np.repeat(np.arange(self.num_nodes), self.num_nodes)
		det = np.tile(np.arange(self.num_nodes), self.num_nodes)
		u, v = (torch.tensor(src), torch.tensor(det))
		g = dgl.graph((u, v))

        # add node features and edge features
		g.ndata["feat"] = torch.from_numpy(node_features)
		g.edata["weight"] = torch.from_numpy(edge_weights)

		return g, label

	def __len__(self):
		return len(self.items_list)

def main():
	dataset = Polar("dataset", "train")
	print(f"Size of set: {len(dataset):d}")
	dataiter = iter(dataset)
	g, label = next(dataiter)
	print(g, label)


if __name__ == "__main__":
	main()
