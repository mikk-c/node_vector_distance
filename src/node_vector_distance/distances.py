import torch
import numpy as np
from . import __device__
from .utils import _Linv_cpu, _Linv_gpu

def _ge_cpu(data, i, j, Linv = None):
   if Linv is None:
      Linv = _Linv_cpu(data["edge_index"])
   diff = data["node_vects"].values[:,i] - data["node_vects"].values[:,j]
   return np.sqrt(diff.T.dot(Linv.dot(diff)))

def _ge_pairwise_cpu(data, Linv = None):
   if Linv is None:
      Linv = _Linv_cpu(data["edge_index"])
   n_vectors = data["node_vects"].shape[1]
   distances = np.zeros((
      n_vectors,
      n_vectors
   ))
   for i in range(n_vectors):
      diff = data["node_vects"].values[:,i] - data["node_vects"].values[:,i + 1:].T
      distances[i,i + 1:] = (diff * Linv.dot(diff.T).T).sum(axis = 1)
   return np.sqrt(distances + distances.T)

def _ge_gpu(tensor, i, j, Linv = None):
   if Linv is None:
      Linv = _Linv_gpu(tensor)
   diff = tensor.node_vects[:,i] - tensor.node_vects[:,j]
   return float(torch.sqrt(diff @ Linv @ diff).cpu())

def _ge_pairwise_gpu(tensor, Linv = None):
   if Linv is None:
      Linv = _Linv_gpu(tensor)
   n_vectors = tensor.node_vects.shape[1]
   distances = torch.zeros((
      n_vectors,
      n_vectors
   )).to(__device__)
   for i in range(n_vectors):
      diff = tensor.node_vects[:,i] - tensor.node_vects[:,i + 1:].T
      distances[i,i + 1:] = (diff * (Linv @ diff.T).T).sum(dim = 1)
   return torch.sqrt(distances + distances.T)

def generalized_euclidean(attr_graph, v1_index, v2_index, Linv = None, workflow = "gpu"):
   if workflow == "gpu":
      return _ge_gpu(attr_graph.data, v1_index, v2_index, Linv = Linv)
   else:
      return _ge_cpu(attr_graph.data, v1_index, v2_index, Linv = Linv)

def pairwise_generalized_euclidean(attr_graph, Linv = None, workflow = "gpu"):
   if workflow == "gpu":
      return _ge_pairwise_gpu(attr_graph.data, Linv = Linv)
   else:
      return _ge_pairwise_cpu(attr_graph.data, Linv = Linv)