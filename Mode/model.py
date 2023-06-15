import torch.nn as nn
import torch
import numpy as np

# d_model = 512  # Embedding Size
# d_ff = 1024  # FeedForward dimension
# d_k = d_v = 64  # dimension of K(=Q), V
# n_layers = 1  # number of Encoder of Decoder Layer
# n_heads = 1  # number of heads in Multi-Head Attention
# batch_size = 8


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class Multi_model(nn.Module):
    def __init__(self, max_ast_node, src_max_length, d_k):
        super(Multi_model, self).__init__()
        # self.Linear = nn.Linear(768, d_model)
        self.conv1 = nn.Conv1d(max_ast_node, max_ast_node, 1, stride=1)
        self.conv2 = nn.Conv1d(src_max_length, max_ast_node, 1, stride=1)
        # self.pooling = nn.
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # self.pos_emb = PositionalEncoding(d_model)
        # 设定encoder的个数
        self.enc_self_attn = ScaledDotProductAttention(d_k)
        # self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, gcn_embed, src_embed, AST_embed):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        # print(gcn_embed)
        # print(src_embed)
        gcn_embed1 = self.conv1(gcn_embed)
        src_embed1 = self.conv2(src_embed)

        # enc_self_attns = []
        enc_outputs, enc_self_attn = self.enc_self_attn(gcn_embed1, src_embed1, src_embed1)
        # for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            # enc_outputs, enc_self_attn = layer(gcn_embed1, src_embed1)
            # enc_outputs = enc_outputs 
            # enc_self_attns.append(enc_self_attn)
        enc_outputs = enc_outputs + AST_embed
        # print(enc_outputs.shape)
        return enc_outputs
