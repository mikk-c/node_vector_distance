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

def network_correlation(attr_graph, v1_index, v2_index, Linv = None, ER = None, W = None, workflow = "gpu"):
   r"""Calculates the correlation of two numeric node attribute vectors over a given network.

   Parameters
   ----------
   attr_graph : :class:`~node_vector_distance.AttrGraph`
      The attributed graph container with the graph in ``attr_graph.data["edges"]`` and the node attributes
      in ``attr_graph.data["node_vects"]``.
   v1_index : int
      The index of the first node vector with which to calculate the correlation.
   v2_index : int
      The index of the second node vector with which to calculate the correlation.
   Linv : :class:`~numpy.ndarray` or :class:`~torch.Tensor`, optional
      The matrix containing the pseudoinverse of the Laplacian of ``attr_graph``. If not provided, and both
      ``W`` and ``ER`` are ``None``, it will be computed automatically but not cached. Should be a
      :class:`~numpy.ndarray` if using ``workflow="cpu"``, or a :class:`~torch.Tensor` if using
      ``workflow="gpu"`` (the default).
   ER : :class:`~numpy.ndarray` or :class:`~torch.Tensor`, optional
      The matrix containing the effective resistance matrix of ``attr_graph``. If not provided, and ``W`` is
      ``None``, it will be computed automatically but not cached. Should be a :class:`~numpy.ndarray` if
      using ``workflow="cpu"``, or a :class:`~torch.Tensor` if using ``workflow="gpu"`` (the default).
   W : :class:`~numpy.ndarray` or :class:`~torch.Tensor`, optional
      The weight matrix for the correlation on ``attr_graph``, defined as ``1 / exp(ER)``. If not provided,
      it will be computed automatically but not cached. Should be a :class:`~numpy.ndarray` if using
      ``workflow="cpu"``, or a :class:`~torch.Tensor` if using ``workflow="gpu"`` (the default).
   workflow : str, optional (default: "gpu")
      Specifies whether to use the torch functions (if equal to "gpu") or the numpy functions (if equal to
      "cpu"). Defaults to "gpu".

   Returns
   -------
   float
      The correlation value, a float number between -1 (perfect negative correlation), and +1 (perfect
      positive correlation) with 0 implying no correlation.

   Raises
   ------
   AttributeError
      If passing an ``attr_graph`` built with a different ``workflow`` than the one specified by the
      parameter.
   TypeError
      If passing a cached ``Linv``, ``ER``, or ``W`` built with a different ``workflow`` than the one
      specified by the parameter.
   IndexError
      If either ``v1_index`` or ``v2_index`` exceed the number of node attributes in ``attr_graph``.

   Notes
   -----
   The network correlation of two vectors :math:`v_1` and :math:`v_2` is defined in [network-pearson]_
   as:

   .. math::
   
      \rho(v_1, v_2, G) = \dfrac{\mathrm{sum}(W \times (\hat{v_1} \otimes \hat{v_2}))}
                                 {\sigma_{v_1,W} \sigma_{v_2,W}},

   where :math:`W` is one over the exponential effective resistance matrix of graph :math:`G`,
   :math:`\hat{v}` is a vector :math:`v` minus its average, :math:`\otimes` indicates the outer product,
   and :math:`\sigma_{v,W} = \sqrt{\mathrm{sum}(W \times (\hat{v} \otimes \hat{v}))}` is a measure of
   network variance of vector :math:`v` over :math:`G`.

   References
   ----------
   .. [network-pearson] Coscia, Michele. "Pearson correlations on complex networks." Journal of Complex
      Networks 9, no. 6 (2021): cnab036. :doi:10.1093/comnet/cnab036
   """
   if workflow == "gpu":
      return _corr_gpu(attr_graph.data, v1_index, v2_index, Linv = Linv, ER = ER, W = W)
   else:
      return _corr_cpu(attr_graph.data, v1_index, v2_index, Linv = Linv, ER = ER, W = W)