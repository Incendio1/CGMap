import networkx as nx
from torch_geometric.utils import (
    add_remaining_self_loops, degree, coalesce, to_undirected
)
from torch_scatter import scatter
import torch
import numpy as np


def edgelist2graph(edge_index, nodenum):
    edge_index = edge_index.cpu().detach().numpy()
    adjlist = {i: [] for i in range(nodenum)}

    for i in range(len(edge_index[0])):
        adjlist[edge_index[0][i]].append(edge_index[1][i])

    return adjlist, nx.adjacency_matrix(nx.from_dict_of_lists(adjlist)).toarray()


def OPP_edge_info(data, args):
    adjlist, adjmatrix = edgelist2graph(data.edge_index, data.x.size(0))
    hop_edge_index, hop_edge_att = OPP(data.x.size(0), args.OPP_layer, adjlist, data.edge_index)
    torch.save(hop_edge_index, './OPP_info/hop_edge_index_' + args.dataset + '_' + str(args.OPP_layer))
    torch.save(hop_edge_att, './OPP_info/hop_edge_att_' + args.dataset + '_' + str(args.OPP_layer))


def _make_undirected_with_self_loops(edge_index, num_nodes):
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


def propagate(x, edge_index):
    row, col = edge_index
    deg = degree(col, x.size(0), dtype=x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    out = edge_weight.view(-1, 1) * x[row]
    return scatter(out, col, dim=0, dim_size=x.size(0), reduce='add')


def weight_deg(nodenum, edge_index, K):
    edge_index = _make_undirected_with_self_loops(edge_index, nodenum)

    att = []
    x = torch.eye(nodenum, dtype=torch.float64)

    for i in range(K):
        x = propagate(x, edge_index).double()
        att.append(x.float())

        print(
            f"[DIAG] P^{i+1}: "
            f"nnz={(x != 0).sum().item():,}, "
            f"|w|max={x.abs().max().item():.4e}, "
            f"|w|mean(nz)={x[x != 0].abs().mean().item() if (x != 0).any() else 0:.4e}, "
            f"|w|min(nz)={x[x != 0].abs().min().item() if (x != 0).any() else 0:.4e}"
        )

    return att


def OPP(nodenum, K, adjlist, edge_index):
    att = weight_deg(nodenum, edge_index, K)

    hop_edge_index = [np.zeros((2, nodenum**2), dtype=int) for _ in range(K)]
    hop_edge_att = [np.zeros(nodenum**2, dtype=np.float64) for _ in range(K)]
    hop_edge_pointer = np.zeros(K, dtype=int)

    for i in range(nodenum):
        hop_edge_index, hop_edge_att, hop_edge_pointer = Width_adjlist(
            adjlist, nodenum, hop_edge_index, hop_edge_att,
            hop_edge_pointer, att, source=i, depth_limit=K
        )

    for i in range(K):
        hop_edge_index[i] = hop_edge_index[i][:, :hop_edge_pointer[i]]
        hop_edge_att[i] = hop_edge_att[i][:hop_edge_pointer[i]]

        raw_max = float(np.max(np.abs(hop_edge_att[i]))) if len(hop_edge_att[i]) > 0 else 0.0
        raw_mean = float(np.mean(np.abs(hop_edge_att[i]))) if len(hop_edge_att[i]) > 0 else 0.0
        raw_nonzero = int(np.count_nonzero(hop_edge_att[i])) if len(hop_edge_att[i]) > 0 else 0

        print(
            f"[OPP] hop={i+1:2d}, "
            f"edges={hop_edge_pointer[i]:,}, "
            f"nonzero={raw_nonzero:,}, "
            f"raw_max={raw_max:.6e}, "
            f"raw_mean={raw_mean:.6e}"
        )

        hop_edge_index[i] = torch.tensor(hop_edge_index[i], dtype=torch.long)
        hop_edge_att[i] = torch.tensor(hop_edge_att[i], dtype=torch.float)

    return hop_edge_index, hop_edge_att


def Width_adjlist(adjlist, nodenum, hop_edge_index, hop_edge_att,
                  hop_edge_pointer, deg_att, source, depth_limit):
    visited = {}
    for node in adjlist.keys():
        visited[node] = 0

    queue, output = [], []
    queue.append(source)
    visited[source] = 1
    level = 1

    for i in range(len(hop_edge_index)):
        hop_edge_index[i][0, hop_edge_pointer[i]] = source
        hop_edge_index[i][1, hop_edge_pointer[i]] = source

    tmp = 0
    for k in range(0, depth_limit):
        tmp += deg_att[k][source, source].item()
    hop_edge_att[0][hop_edge_pointer[0]] = tmp
    hop_edge_pointer[0] += 1

    while queue:
        level_size = len(queue)
        while level_size != 0:
            vertex = queue.pop(0)
            level_size -= 1
            for vrtx in adjlist[vertex]:
                if visited[vrtx] == 0:
                    queue.append(vrtx)
                    visited[vrtx] = 1

                    hop_edge_index[level - 1][0, hop_edge_pointer[level - 1]] = source
                    hop_edge_index[level - 1][1, hop_edge_pointer[level - 1]] = vrtx

                    tmp = 0
                    for k in range((level - 1), depth_limit):
                        tmp += deg_att[k][vrtx, source].item()
                    hop_edge_att[level - 1][hop_edge_pointer[level - 1]] = tmp
                    hop_edge_pointer[level - 1] += 1

        level += 1
        if level > depth_limit:
            break

    return hop_edge_index, hop_edge_att, hop_edge_pointer