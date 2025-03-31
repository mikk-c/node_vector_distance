import torch, torch_geometric
import numpy as np
from . import __device__
from .utils import spl_matrix, _er_cpu, _er_gpu

def _variance_cpu(data, i, dists = None, n_proc = 1, kernel = "resistance"):
   v = data["node_vects"].values[:,i] / data["node_vects"].values[:,i].sum()
   if kernel == "geodesic":
      if dists is None:
         dists = spl_matrix(data["edge_index"], n_proc = n_proc)
   elif kernel == "resistance":
      if dists is None:
         dists = _er_cpu(data["edge_index"])
   return (np.outer(v, v) * (dists ** 2)).sum() / 2

def _variance_gpu(tensor, i, dists = None, n_proc = 1, kernel = "resistance"):
   v = tensor.node_vects[:,i] / tensor.node_vects[:,i].sum()
   if kernel == "geodesic":
      if dists is None:
         G = torch_geometric.utils.to_networkx(
            tensor,
            to_undirected = True,
            edge_attrs = ["edge_weights",]
         )
         dists = spl_matrix(G, n_proc = n_proc)
      dists = torch.tensor(dists).double().to(__device__)
   elif kernel == "resistance":
      if dists is None:
         dists = _er_gpu(tensor)
   return float(((torch.outer(v, v) * (dists ** 2)).sum() / 2).cpu())

def variance(attr_graph, v_index, dists = None, n_proc = 1, kernel = "resistance", workflow = "gpu"):
   if workflow == "gpu":
      return _variance_gpu(attr_graph.data, v_index, dists = dists, n_proc = n_proc, kernel = kernel)
   else:
      return _variance_cpu(attr_graph.data, v_index, dists = dists, n_proc = n_proc, kernel = kernel)