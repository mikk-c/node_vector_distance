"""
``node_vector_distance``
------------------------

This is the core module providing the fundamental data structures and functions.

Fundamental classes
+++++++++++++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   AttrGraph

Functions
+++++++++

.. autosummary::
   :nosignatures:
   :toctree: autosummary

   _make_tensor
   _make_graph
"""

__author__ = "Michele Coscia <mcos@itu.dk>"

import torch

__version__ = "0.0.21"
__device__ = "cuda" if torch.cuda.is_available() else "cpu"

from .utils import *
from .variance import *
from .distances import *
from .correlation import *

import torch_geometric
import numpy as np
import networkx as nx

class AttrGraph(object):
   """General graph class.

   This class holds a graph and all of its node and edge attributes. It can only be built
   starting from a :class:`networkx.Graph` object `G` and a :class:`pandas.Dataframe` `df`.
   The :class:`networkx.Graph` must be undirected, none of the functions provided in this
   library are defined for directed graphs. The :class:`networkx.Graph` can have edge
   weights, which must be stored in the `weight` edge attribute of `G`.

   The graph data (nodes, edges, and all of their attributes), will be sored in
   `AttrGraph.data` with this structure:

      1. `data["edge_index"]` contains the edges.

      2. `data["node_vects"]` contains the node attributes. It is a matrix with one node
         per row and one attribute per column. The rows are sorted so that the first row
         refers to the node with id 0, the second row to node with id 1, and so on. The
         columns have the same order as the `df` parameter.

      3. `data["edge_weights"]` contains a sequence of edge weights, in the same order as
         the edges in `data["edge_index"]` (so the n-th element of this sequence is the
         weight of the n-th edge in `data["edge_index"]`).

      4. `data["edge_attrs"]` contains a matrix of edge attributes. Each row is an edge,
         in the same order as `data["edge_index"]` and `data["edge_weights"]`, and each
         column is an edge attribute, with the same order as specified in `edge_attr_order`.

   You can specify the order in which the edge attributes should be stored, which is useful
   if you use a GPU workflow and therefore will work with a
   :class:`torch_geometric.data.Data`. This can be done by passing the list of attributes
   names as the `edge_attr_order` optional parameter. If not passed, it will default with
   whatever order edge attributes have in the first edge in `G`.

   To specify which workflow you want to use, you can set the optional `workflow` parameter
   to be either `"gpu"` (the default), or `"cpu"`. If `workflow="gpu"` (the default) then
   `AttrGraph.data` is actually a :class:`torch_geometric.data.Data` object. If
   `workflow="cpu"`, then `AttrGraph.data` is a dictionary containing `G` and `df`.

   Since ``NVD`` methods only work on networks with a single connected component, and
   ``torch_geometric`` requires numeric ids without gaps, `AttrGraph.data` will only
   contain the nodes and edges in the largest connected component of `G`. Moreover, the
   node ids of `G` will be changed so that they start from `0`to `n` without gaps. For
   your convenience, `AttrGraph.nodemap` contains a dictionary that maps the node ids
   in `G`with the node ids in `AttrGraph.data`.
   """
   def __init__(self, G, df, edge_attr_order = None, workflow = "gpu"):
      if not type(G) is nx.Graph:
         raise ValueError("""
            G is not a networkx Graph.
         """)
      self.workflow = workflow
      G, df, self.nodemap = _extract_lcc(G, df)
      edge_attrs = next(iter(G.edges(data = True)))[2]
      if "weight" in edge_attrs:
         edge_weights = []
      else:
         edge_weights = [1.] * len(G.edges) * 2
      if edge_attr_order is None:
         edge_attr_order = [attr for attr in edge_attrs if attr != "weight"]
      if workflow == "gpu":
         self.data = _make_tensor(G, df, edge_attr_order, edge_weights)
      else:
         self.data = _make_graph(G, df, edge_attr_order, edge_weights)

   def update_node_vects(self, df):
      r"""Updates the graph's node vectors.

      Parameters
      ----------
      df : :class:`~pandas.DataFrame`
         A :class:`~pandas.DataFrame` containing the node attributes. Each attribute is one column
         of the data frame and each row is a node. The nodes must be sorted coherently with the node
         ids of the :class:`~node_vector_distance.AttrGraph` otherwise the distances calculated won't
         make sense. So the first       row must correspond with node with id `0`m the second row
         with node id `1`, and so on.
      """
      if self.workflow == "gpu":
         self.data.node_vects = torch.tensor(df.values).float().to(__device__)
      else:
         self.data["node_vects"] = df

################################################################################
# Utility functions
################################################################################

def _make_tensor(G, df, edge_attr_order, edge_weights):
   r"""Creates a :class:`~torch_geometric.data.Data` that can be used as the data container to
   calculate network distances. This can be useful if you already preprocessed your data and don't
   need to build an :class:`~node_vector_distance.AttrGraph` with its default constructor. However,
   this skips a lot of checks so it might be tricky to use, see Notes.

   Parameters
   ----------
   G : :class:`~networkx.Graph`
      The object containing the edges of the graph with their attributes.
   df : :class:`~pandas.DataFrame`
      The object containing the node attributes of the graph.
   edge_attr_order : list
      The list containing the order in which you want to store the edge attributes in the tensor.
      Can be an empty list if the graph has no edge attributes. Do not use this for edge weights.
   edge_weights : list
      The list containing the edge weights. If empty, the object won't be created properly. If your
      graph is unweighted, you must pass a list of ones.

   Returns
   -------
   :class:`~torch_geometric.data.Data`
      The data frame containing all the information in your graph.

   Notes
   -----
   The returned object cannot be used "as is" with the functions of this library. It needs to be
   wrapped as the "data" key in a dictionary.

   Note that all functions assume that the graph has a single connected component. This is enforced
   when creating a standard :class:`~node_vector_distance.AttrGraph`, but it is not checked here.
   Passing a graph with multiple connected components will cause issues.

   Creating a standard :class:`~node_vector_distance.AttrGraph` also enforces numeric node ids
   without gaps. Also this is not enforced here. Providing non-numeric node ids and/or ids with
   gaps will cause issues. For instance, id gaps will be interpreted as isolated nodes.

   By using this function, you won't have an `AttrGraph.nodemap` attribute, so it is up to you to
   keep track of which node has which id.
   """
   edge_index = [[], []]
   edge_attr = []
   hasweights = len(edge_weights) == 0
   hasattrs = len(edge_attr_order) > 0
   for edge in G.edges(data = True):
      edge_index[0].append(edge[0])
      edge_index[1].append(edge[1])
      edge_index[0].append(edge[1])
      edge_index[1].append(edge[0])
      if hasweights:
         edge_weights.append(edge[2]["weight"])
         edge_weights.append(edge[2]["weight"])
      if hasattrs:
         edge_attr.append([edge[2][attr] for attr in edge_attr_order])
         edge_attr.append([edge[2][attr] for attr in edge_attr_order])
   tensor = torch_geometric.data.Data(
      edge_index = torch.tensor(edge_index).to(__device__),
      node_vects = torch.tensor(df.values).float().to(__device__),
      edge_weights = torch.tensor(edge_weights).to(__device__),
      edge_attr = torch.tensor(edge_attr).to(__device__)
   )
   return tensor

def _make_graph(G, df, edge_attr_order, edge_weights):
   r"""Creates a dictionary that can be used as the data container to  calculate network distances.
   This can be useful if you already preprocessed your data and don't need to build an
   :class:`~node_vector_distance.AttrGraph` with its default constructor. However, this skips a lot
   of checks so it might be tricky to use, see Notes.

   Parameters
   ----------
   G : :class:`~networkx.Graph`
      The object containing the edges of the graph with their attributes.
   df : :class:`~pandas.DataFrame`
      The object containing the node attributes of the graph.
   edge_attr_order : list
      The list containing the order in which you want to store the edge attributes in the tensor.
      Can be an empty list if the graph has no edge attributes. Do not use this for edge weights.
   edge_weights : list
      The list containing the edge weights. If empty, the object won't be created properly. If your
      graph is unweighted, you must pass a list of ones.

   Returns
   -------
   dict
      The dictionary containing all the information in your graph.

   Notes
   -----
   The returned object cannot be used "as is" with the functions of this library. It needs to be
   wrapped as the "data" key in a dictionary.

   Note that all functions assume that the graph has a single connected component. This is enforced
   when creating a standard :class:`~node_vector_distance.AttrGraph`, but it is not checked here.
   Passing a graph with multiple connected components will cause issues.

   Creating a standard :class:`~node_vector_distance.AttrGraph` also enforces numeric node ids
   without gaps. Also this is not enforced here. Providing non-numeric node ids and/or ids with
   gaps will cause issues. For instance, id gaps will be interpreted as isolated nodes.

   By using this function, you won't have an `AttrGraph.nodemap` attribute, so it is up to you to
   keep track of which node has which id.
   """
   data = {}
   data["edge_index"] = G
   data["node_vects"] = df
   data["edge_weights"] = np.array([e[2]["weight"] for e in G.edges(data = True)]) if len(edge_weights) == 0 else edge_weights[::2]
   data["edge_attr"] = np.array([[e[2][attr] for attr in edge_attr_order] for e in G.edges(data = True)])
   return data

#### Making sure G has the correct node ids (from 0 to n without gaps)
def _convert_to_int_node_ids(G):
   nodemap = {list(G.nodes)[i]: i for i in range(len(G.nodes))}
   nodemap_reverse = {nodemap[n]: n for n in nodemap}
   G = nx.relabel_nodes(G, nodemap)
   H = nx.Graph()
   H.add_nodes_from(sorted(G.nodes))
   H.add_edges_from(G.edges)
   nx.set_node_attributes(H, nodemap_reverse, "original_id")
   return H, nodemap_reverse

#### Reorder the node attribute dataframe after node ids have been cleaned
def _consolidate_node_attributes(df, nodemap):
   df.index = df.index.map(nodemap)
   return df.sort_index()

#### Making sure G has a single connected component
def _extract_lcc(G, df):
   lcc = max(nx.connected_components(G), key = len)
   G = G.subgraph(lcc).copy()
   G, nodemap = _convert_to_int_node_ids(G)
   df = _consolidate_node_attributes(df, nodemap)
   return G, df, nodemap