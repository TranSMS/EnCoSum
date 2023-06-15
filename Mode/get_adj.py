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

    for ast in papers:
        ast[0].update({'layer': 0})
        for a in ast:
            if 'children' in a:
                for i in a['children']:
                    ast[i].update({'layer': a['layer'] + 1})
        # print(ast)
        id_ch = {t['id']: t['children'] for t in ast if 'children' in t}
        # print(id_ch)
        # exit()
        edgelist = []
        for id in id_ch:
            for child in id_ch[id]:
                # fo.write(str(id)+'\t'+str(child)+'\n')
                edgelist.append((id, child))
        #
        id_type_for = {d['id']: d['children'] for d in ast if d['type'] == 'ForStatement' and 'children' in d}
        id_type_block = {w['id']: w['children'] for w in ast if w['type'] == 'BlockStatement' and 'children' in w}
        id_type_while = {n['id']: n['children'] for n in ast if n['type'] == 'WhileStatement' and 'children' in n}
        id_type_if = {m['id']: m['children'] for m in ast if m['type'] == 'IfStatement' and 'children' in m}
        non = []
        for b in ast:
            if "children" in b:
                non.append(b["id"])

        edgelist = []
        for id in id_ch:
            for child in id_ch[id]:
                edgelist.append((id, child))
        value_list_for = id_type_for.values()
        for_list = list(value_list_for)
        # 遍历每个字典
        classified_dict = {}
        for i, d in enumerate(ast):
            layer = d['layer']
            if layer in classified_dict:
                classified_dict[layer].append(i)
            else:
                classified_dict[layer] = [i]
        # print(classified_dict)
        add_nub_edges = []
        for value in classified_dict.values():
            # print(value)
            if len(value) > 2:
                for i in range(len(value) - 1):
                    xx = (value[i], value[i + 1])
                    add_nub_edges.append(xx)
        for k in add_nub_edges:
            edgelist.append(k)
        add_non_edges = []
        for i in range(len(non) - 1):
            x = (non[i], non[i+1])
            add_non_edges.append(x)
        for j in add_non_edges:
            edgelist.append(j)
        add_for_edges = []
        for v in for_list:
            for_list2 = v
            if len(for_list2) > 0:
                add_for_edges.append((for_list2[0], for_list2[-1]))

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
        # print(aa2)

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
