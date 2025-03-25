import torch, torch_geometric
import networkx as nx
from . import __device__

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

#### Creating a torch tensor from a networkx graph and a pandas dataframe with a collection of node attributes
def _make_tensor(G, df, edge_attr_order = None):
   if not G.has_edge(0, 1):
      raise ValueError("""
         The Graph doesn't have edge 0,1. It was likely badly constructed.
      """)
   edge_index = [[], []]
   if "weight" in G[0][1]:
      edge_weights = []
      hasweights = True
   else:
      edge_weights = [1.] * len(G.edges) * 2
      hasweights = False
   if edge_attr_order is None:
      edge_attr_order = [attr for attr in G[0][1] if attr != "weight"]
   hasattrs = len(edge_attr_order) > 0
   edge_attr = []
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

def _make_graph(G, df):
   data = {}
   data["edge_index"] = G
   data["node_vects"] = df
   return data

def build_data(G, df, edge_attr_order = None, method = "gpu"):
   G, df, nodemap = _extract_lcc(G, df)
   if method == "gpu":
      return _make_tensor(G, df, edge_attr_order = edge_attr_order), nodemap
   else:
      return _make_graph(G, df), nodemap