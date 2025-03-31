import torch, torch_geometric
import numpy as np
import networkx as nx
from . import __device__

class AttrGraph(object):
   def __init__(self, G, df, edge_attr_order = None, workflow = "gpu"):
      G, df, self.nodemap = _extract_lcc(G, df)
      if not G.has_edge(0, 1):
         raise ValueError("""
            The Graph doesn't have edge 0,1. It was likely badly constructed.
         """)
      if "weight" in G[0][1]:
         edge_weights = []
      else:
         edge_weights = [1.] * len(G.edges) * 2
      if edge_attr_order is None:
         edge_attr_order = [attr for attr in G[0][1] if attr != "weight"]
      if workflow == "gpu":
         self.data = _make_tensor(G, df, edge_attr_order, edge_weights)
      else:
         self.data = _make_graph(G, df, edge_attr_order, edge_weights)

def _make_tensor(G, df, edge_attr_order, edge_weights):
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
   data = {}
   data["edge_index"] = G
   data["node_vects"] = df
   data["edge_weights"] = np.array([e[2]["weight"] for e in G.edges(data = True)]) if len(edge_weights) == 0 else edge_weights[::2]
   data["edge_attrs"] = np.array([[e[2][attr] for attr in edge_attr_order] for e in G.edges(data = True)])
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