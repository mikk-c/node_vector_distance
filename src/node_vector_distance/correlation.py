import torch
import numpy as np
from .utils import _Linv_cpu, _Linv_gpu, _er_cpu, _er_gpu

def _corr_cpu(data, i, j, Linv = None, ER = None, W = None):
   if W is None:
      if ER is None:
         if Linv is None:
            Linv = _Linv_cpu(data["edge_index"])
         ER = _er_cpu(data["edge_index"], Linv = Linv)
      W = 1 / np.exp(ER)
   v1_hat = data["node_vects"].values[:,i] - data["node_vects"].values[:,i].mean()
   v2_hat = data["node_vects"].values[:,j] - data["node_vects"].values[:,j].mean()
   numerator = (W * np.outer(v1_hat, v2_hat)).sum()
   denominator_v1 = np.sqrt((W * np.outer(v1_hat, v1_hat)).sum())
   denominator_v2 = np.sqrt((W * np.outer(v2_hat, v2_hat)).sum())
   return numerator / (denominator_v1 * denominator_v2)

def _corr_gpu(tensor, i, j, Linv = None, ER = None, W = None):
   if W is None:
      if ER is None:
         if Linv is None:
            Linv = _Linv_gpu(tensor)
         ER = _er_gpu(tensor, Linv = Linv)
      W = 1 / torch.exp(ER)
   i_hat = tensor.node_vects[:,i] - tensor.node_vects[:,i].mean()
   j_hat = tensor.node_vects[:,j] - tensor.node_vects[:,j].mean()
   numerator = (W * torch.outer(i_hat, j_hat)).sum()
   denominator_i = torch.sqrt((W * torch.outer(i_hat, i_hat)).sum())
   denominator_j = torch.sqrt((W * torch.outer(j_hat, j_hat)).sum())
   return float((numerator / (denominator_i * denominator_j)).cpu())

def network_correlation(data, v1_index, v2_index, Linv = None, ER = None, W = None, method = "gpu"):
   """Calculates the correlation of two numeric node attribute vectors over a given network.
   :param data: The data container containing the graph in data["edges"] and the node attributes in data["node_vects"]
   :type data: [ParamType](, optional)
   ...
   :raises [ErrorType]: [ErrorDescription]
   ...
   :return: [ReturnDescription]
   :rtype: [ReturnType]
   """
   if method == "gpu":
      return _corr_gpu(data, v1_index, v2_index, Linv = Linv, ER = ER, W = W)
   else:
      return _corr_cpu(data, v1_index, v2_index, Linv = Linv, ER = ER, W = W)