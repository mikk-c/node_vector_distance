import torch, torch_geometric
import numpy as np
import networkx as nx
from scipy.sparse import csgraph
from multiprocessing import Pool, Manager

#### Calculate the pseudoinverse of the Laplacian (CPU numpy version)
def _Linv_cpu(G):
   A = nx.adjacency_matrix(G).todense().astype(float)
   return np.linalg.pinv(csgraph.laplacian(np.matrix(A), normed = False))

#### Calculate the pseudoinverse of the Laplacian (GPU torch version)
def _Linv_gpu(tensor):
   L_ei, Lew = torch_geometric.utils.get_laplacian(
      tensor.edge_index,
      edge_weight = tensor.edge_weights
   )
   L = torch_geometric.utils.to_dense_adj(edge_index = L_ei, edge_attr = Lew)[0]
   return torch.linalg.pinv(L, hermitian = True)

def laplacian_pseudoinverse(attr_graph, workflow = "gpu"):
   if workflow == "gpu":
      return _Linv_gpu(attr_graph.data)
   else:
      return _Linv_cpu(attr_graph.data["edge_index"])

#### Calculate a matrix of all shortest path lengths
def _spl(x):
   x[2][x[1]] = dict(
      nx.shortest_path_length(
         x[0],
         source = x[1],
         weight = "weight"
      )
   )

def spl_matrix(edge_index, nodes = None, n_proc = 1):
   if nodes is None:
      nodes = list(range(len(edge_index.nodes)))
   manager = Manager()
   spls = manager.dict()
   pool = Pool(processes = n_proc)
   _ = pool.map(
      _spl,
      [(edge_index, n, spls) for n in nodes]
   )
   pool.close()
   pool.join()
   spls = dict(spls)
   spls = [
      [spls[i][j] if j in spls[i] else np.inf for i in nodes] for j in nodes
   ]
   return np.array(spls)

#### Get the Effective Resistance Matrix (CPU numpy version)
def _er_cpu(G, Linv = None):
   if Linv is None:
      Linv = _Linv_cpu(G)
   z = np.diag(Linv)
   u = np.ones(z.shape[0])
   return np.array((np.matrix(u).T * z) + (np.matrix(z).T * u) - (2 * Linv))

#### Get the Effective Resistance Matrix (GPU torch version)
def _er_gpu(tensor, Linv = None):
   if Linv is None:
      Linv = _Linv_gpu(tensor)
   pinv_diagonal = torch.diagonal(Linv)
   return pinv_diagonal.unsqueeze(0) +  pinv_diagonal.unsqueeze(1) - 2 * Linv

def effective_resistance(attr_graph, workflow = "gpu", Linv = None):
   if workflow == "gpu":
      return _er_gpu(attr_graph.data, Linv = Linv)
   else:
      return _er_cpu(attr_graph.data["edge_index"], Linv = Linv)