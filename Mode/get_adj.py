import json
import networkx as nx
import numpy as np
import torch
import scipy.sparse as sp
from scipy import sparse


def read_batchA(ast_file, max_node):
    file = open(ast_file, 'r', encoding='utf-8')
    papers = []
    a1 = []
    aa2 = []
    aa3 = []
    aa4 = []
    aa5 = []

    for line in file.readlines():
        dic = json.loads(line)

        papers.append(dic)
        id_ch = {t['id']: t['children'] for t in dic if 'children' in t}
        # print(id_ch)
        # exit()
        edgelist = []
        for id in id_ch:
            for child in id_ch[id]:
                # fo.write(str(id)+'\t'+str(child)+'\n')
                edgelist.append((id, child))
        # print(edgelist)
        #
        id_type_for = {d['id']: d['children'] for d in dic if d['type'] == 'ForStatement' and 'children' in d}
        id_type_block = {w['id']: w['children'] for w in dic if w['type'] == 'BlockStatement' and 'children' in w}
        id_type_while = {n['id']: n['children'] for n in dic if n['type'] == 'WhileStatement' and 'children' in n}
        id_type_if = {m['id']: m['children'] for m in dic if m['type'] == 'IfStatement' and 'children' in m}
        # print(id_type_for)
        # print(id_type_block)
        # print(id_type_while)
        # print(id_type_if)

        edgelist = []
        for id in id_ch:
            for child in id_ch[id]:
                # fo.write(str(id)+'\t'+str(child)+'\n')
                edgelist.append((id, child))
        # print(edgelist)

        value_list_for = id_type_for.values()
        for_list = list(value_list_for)
        add_for_edges = []
        for v in for_list:
            for_list2 = v
            if len(for_list2) > 0:
                add_for_edges.append((for_list2[0], for_list2[-1]))

        # print(add_edges)
        # print(add_for_edges)
        for i in add_for_edges:
            edgelist.append(i)

        value_list_block = id_type_block.values()
        block_list = list(value_list_block)
        # print(block_list)
        add_block_edges = []
        for b in block_list:
            block_list2 = b
            # print(block_list2)
            if len(block_list2) > 1:
                for k in range(len(block_list2) - 1):
                    # print(k)
                    add_block_edges.append((block_list2[k], block_list2[k + 1]))

        # print(add_block_edges)
        for j in add_block_edges:
            edgelist.append(j)

        value_list_while = id_type_while.values()
        while_list = list(value_list_while)
        add_while_edges = []
        if len(while_list) > 0:

            while_list2 = while_list[0]

            if len(while_list) > 0:
                add_while_edges.append((while_list2[0], while_list2[-1]))
        # print(add_while_edges)
        for p in add_while_edges:
            edgelist.append(p)

        value_list_if = id_type_if.values()
        if_list = list(value_list_if)
        add_if_edges = []
        for g in if_list:
            if len(if_list) > 0:

                if_list2 = g

                if len(if_list2) > 0:
                    if len(if_list2) == 3:
                        for c in range(len(if_list2)-1):
                            add_if_edges.append((if_list2[0], if_list2[c + 1]))
                    else:
                        add_if_edges.append((if_list2[0], if_list2[-1]))
        # print(add_if_edges)
        for m in add_if_edges:
            edgelist.append(m)


        G = nx.Graph()
        G.add_edges_from(edgelist)

        nx.draw(G, with_labels=True)
        A = np.array(nx.adjacency_matrix(G).todense())
        A1 = A + sp.eye(A.shape[0])
        A = np.array(A1, dtype=int)
        # print(A)
        if len(A[0]) > max_node:
            a = A[0:max_node, 0:max_node]
            # print(aa)
        else:
            a = np.zeros((max_node, max_node), dtype=int)
            # A = A + a
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    a[i][j] = A[i][j]
        a2 = a.dot(a)
        # print(a2)
        a3 = a.dot(a2)
        a4 = a.dot(a3)
        a5 = a.dot(a4)


        A2 = normalize_data(a2)
        A2 = np.array(A2)
        A2 = sparse.csr_matrix(A2)
        A2 = torch.FloatTensor(np.array(A2.todense()))
        # A2 = sparse_mx_to_torch_sparse_tensor(A2)
        aa2.append(A2)

        A3 = normalize_data(a3)
        A3 = np.array(A3)
        A3 = sparse.csr_matrix(A3)
        A3 = torch.FloatTensor(np.array(A3.todense()))
        aa3.append(A3)

        A4 = normalize_data(a4)
        A4 = np.array(A4)
        A4 = sparse.csr_matrix(A4)
        A4 = torch.FloatTensor(np.array(A4.todense()))
        aa4.append(A4)

        A5 = normalize_data(a5)
        A5 = np.array(A5)
        A5 = sparse.csr_matrix(A5)
        A5 = torch.FloatTensor(np.array(A5.todense()))
        aa5.append(A5)

        a = np.array(a, dtype=float)
        adj = normalize(a)
        # print(adj)
        adj = sp.csr_matrix(adj)
        adj = torch.FloatTensor(np.array(adj.todense()))
        # print(adj)
        a1.append(adj)
    # print(len(a1))

    return a1, aa2, aa3, aa4, aa5


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_data(data):
    list = []
    for line in data:
        q = np.sum(line ** 2)
        # print(q)
        if q != 0:
            normalized_line = line / np.sqrt(q)
            list.append(normalized_line)
        else:
            list.append(line)
    return list
