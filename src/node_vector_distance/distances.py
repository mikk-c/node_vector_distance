"""
``distances``
---------------

This module provides the attribute distance functions.

Summary
+++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   generalized_euclidean
   pairwise_generalized_euclidean

"""

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
   r"""Calculates the generalized euclidean distance of two numeric node attribute vectors over a given
   network.

   Parameters
   ----------
   attr_graph : :class:`~node_vector_distance.AttrGraph`
      The attributed graph container with the graph in ``attr_graph.data["edges"]`` and the node attributes
      in ``attr_graph.data["node_vects"]``.
   v1_index : int
      The index of the first node vector with which to calculate the distance.
   v2_index : int
      The index of the second node vector with which to calculate the distance.
   Linv : :class:`~numpy.ndarray` or :class:`~torch.Tensor`, optional
      The matrix containing the pseudoinverse of the Laplacian of ``attr_graph``. If not provided`, it will
      be computed automatically but not cached. Should be a :class:`~numpy.ndarray` if using
      ``workflow="cpu"``, or a :class:`~torch.Tensor` if using ``workflow="gpu"`` (the default).
   workflow : str, optional (default: "gpu")
      Specifies whether to use the torch functions (if equal to "gpu") or the numpy functions (if equal to
      "cpu"). Defaults to "gpu".

   Returns
   -------
   float
      The distance value, a float number from 0 (the two vectors are the same) to an arbitrarily large
      number indicating the distance.

   Raises
   ------
   AttributeError
      If passing an ``attr_graph`` built with a different ``workflow`` than the one specified by the
      parameter.
   TypeError
      If passing a cached ``Linv`` built with a different ``workflow`` than the one specified by the
      parameter.
   IndexError
      If either ``v1_index`` or ``v2_index`` exceed the number of node attributes in ``attr_graph``.

   Notes
   -----
   The generalized euclidean distance of two vectors :math:`v_1` and :math:`v_2` is defined in
   [generalized-euclidean]_ as:

   .. math::
   
      \delta(v_1, v_2, G) = \sqrt{(v_1 - v_2)^T L^\dagger (v_1 - v_2))},

   where :math:`L^\dagger` is the Moore-Penrose pseudoinverse of the Laplacian of graph :math:`G`, and
   :math:`v` is a numerical vector with one value per node.

   References
   ----------
   .. [generalized-euclidean] Coscia, Michele. "Generalized Euclidean measure to estimate network
      distances." In Proceedings of the international AAAI conference on web and social media, vol. 14,
      pp. 119-129. 2020. :doi:10.1609/icwsm.v14i1.7284 
   """
   if workflow == "gpu":
      return _ge_gpu(attr_graph.data, v1_index, v2_index, Linv = Linv)
   else:
      return _ge_cpu(attr_graph.data, v1_index, v2_index, Linv = Linv)

def pairwise_generalized_euclidean(attr_graph, Linv = None, workflow = "gpu"):
   r"""Calculates the pairwise generalized euclidean distance for all pairs of node attributes in the graph.
   It is more efficient than a nested loop over all possible node attribute pairs.

   Parameters
   ----------
   attr_graph : :class:`~node_vector_distance.AttrGraph`
      The attributed graph container with the graph in ``attr_graph.data["edges"]`` and the node attributes
      in ``attr_graph.data["node_vects"]``.
   Linv : :class:`~numpy.ndarray` or :class:`~torch.Tensor`, optional
      The matrix containing the pseudoinverse of the Laplacian of ``attr_graph``. If not provided`, it will
      be computed automatically but not cached. Should be a :class:`~numpy.ndarray` if using
      ``workflow="cpu"``, or a :class:`~torch.Tensor` if using ``workflow="gpu"`` (the default).
   workflow : str, optional (default: "gpu")
      Specifies whether to use the torch functions (if equal to "gpu") or the numpy functions (if equal to
      "cpu"). Defaults to "gpu".

   Returns
   -------
   :class:`~numpy.ndarray` or :class:`~torch.Tensor`
      The distance matrix, with one row/column per node attribute. The order is the same as the node vectors
      in ``attr_graph``. Each cell contains the distance between the corresponding node attributes. The
      return type depends on the value of the ``workflow`` parameter: :class:`~torch.Tensor` if using
      ``workflow="gpu"``, :class:`~numpy.ndarray` if using ``workflow="cpu"``.

   Raises
   ------
   AttributeError
      If passing an ``attr_graph`` built with a different ``workflow`` than the one specified by the
      parameter.
   TypeError
      If passing a cached ``Linv`` built with a different ``workflow`` than the one specified by the
      parameter.
   """
   if workflow == "gpu":
      return _ge_pairwise_gpu(attr_graph.data, Linv = Linv)
   else:
      return _ge_pairwise_cpu(attr_graph.data, Linv = Linv)