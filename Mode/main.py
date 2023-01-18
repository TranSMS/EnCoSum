import argparse
import time
import math
from visdom import Visdom
# from util import epoch_time
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import nltk
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, random_split
import torch.utils.data as Data
from get_A import read_batchA
from get_embed import get_embed
from util import epoch_time
from MyDataSet import MySet, MySampler
from gcn_model import AST_Model, GCNEncoder
# from transformer2 import Transformer2
from Model import Model
from Train_eval import train, evaluate
from make_data import load_nl_data, load_code_data
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
batch_size = 32
epoches = 180
nl_max_len = 30
# seq_max_len = 111
train_num = 69708  # 960
max_ast_node = 80  # 60
src_max_length = 300  # 120
md_max_len = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=150, help="number of epochs")
parser.add_argument("-sn", "--max_ast_node_num", type=int, default=23, help="AST graph max node number")
parser.add_argument("-s", "--src_max_len", type=int, default=65, help="maximum sequence len")
parser.add_argument("-n", "--nl_max_len", type=int, default=16, help="maximum nl len")
parser.add_argument("-md", "--md_max_len", type=int, default=4, help="maximum method len")
parser.add_argument("-dp", "--dropout", type=float, default=0.1, help="maximum sequence len")
parser.add_argument("-fd", "--nfeat_dim", type=int, default=768, help="graph hidden dimension")
parser.add_argument("-hd", "--nhid_dim", type=int, default=768, help="graph hidden dimension")
parser.add_argument("-od", "--nout_dim", type=int, default=768, help="graph hidden dimension")
parser.add_argument("-l", "--layers", type=int, default=6, help="number of layers")
parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate of adam")
parser.add_argument("-k", "--dk", type=int, default=64, help="transfrmer dimension")
parser.add_argument("-v", "--dv", type=int, default=64, help="Transfrmer dimension")
parser.add_argument("-ff", "--dff", type=int, default=2048, help="Transfrmer dimension")
parser.add_argument("-m", "--dmodel", type=int, default=512, help="Transfrmer dimension")

parser.add_argument("-tn", "--train_num", type=int, default=68, help="training number")

args = parser.parse_args()

tgt_vocab_size, tgt_inv_vocab_dict, dec_inputs, tgt_vocab, dec_outputs = load_nl_data('dataset/example_nl.txt', nl_max_len=args.nl_max_len)
src_vocab_size, enc_inputs, md_inputs, src_vocab = load_code_data('dataset/example_code.txt', 'dataset/example_method.txt', args.src_max_len, args.md_max_len)


A, A2, A3, A4, A5 = read_batchA('dataset/example_ast.txt', args.max_ast_node_num)
X = get_embed('dataset/example_ast.txt', args.max_ast_node_num)

A_train = A[0:args.train_num]
A_test = A[args.train_num:len(A)]

A2_train = A2[0:args.train_num]
A2_test = A2[args.train_num:len(A2)]
A3_train = A3[0:args.train_num]
A3_test = A3[args.train_num:len(A3)]
A4_train = A4[0:args.train_num]
A4_test = A4[args.train_num:len(A4)]
A5_train = A5[0:args.train_num]
A5_test = A5[args.train_num:len(A5)]

X_train = X[0:args.train_num]
X_test = X[args.train_num:len(X)]

enc_inputs = torch.LongTensor(enc_inputs)
md_inputs = torch.LongTensor(md_inputs)
dec_inputs = torch.LongTensor(dec_inputs)
dec_outputs = torch.LongTensor(dec_outputs)

enc_train = enc_inputs[:args.train_num]
enc_test = enc_inputs[args.train_num:]
md_train = md_inputs[:args.train_num]
md_test = md_inputs[args.train_num:]
dec_in_train = dec_inputs[:args.train_num]
dec_in_test = dec_inputs[args.train_num:]
dec_out_train = dec_outputs[:args.train_num]
dec_out_test = dec_outputs[args.train_num:]

train_data = MySet(A_train, X_train, A2_train, A3_train, A4_train, A5_train, enc_train, md_train, dec_in_train, dec_out_train)
evl_data = MySet(A_test, X_test, A2_test, A3_test, A4_test, A5_test, enc_test, md_test, dec_in_test, dec_out_test)

my_sampler1 = MySampler(train_data, args.batch_size)
my_sampler2 = MySampler(evl_data, args.batch_size)
evl_data_loader = DataLoader(evl_data, batch_sampler=my_sampler2)
train_data_loader = DataLoader(train_data, batch_sampler=my_sampler1)

# trans_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=batch_size, shuffle=True)
# gcn_model = GCNEncoder().to(device)
model = Model(src_vocab_size, tgt_vocab_size, max_ast_node=args.max_ast_node_num, src_max_length=args.src_max_len,
                          nfeat=args.nfeat_dim, nhid=args.nhid_dim, nout=args.nout_dim, d_model=args.dmodel, batch_size=args.batch_size, dropout=args.dropout,
                          d_k=args.dk, d_v=args.dv, d_ff=args.dff, n_heads=args.attn_heads, n_layers=args.layers, device=device).to(device)
# trans2_model = Transformer2(src_vocab_size, tgt_vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
LEARNING_RATE = args.lr
N_EPOCHS = args.epochs
# gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=0.0001, momentum=0.99)
model_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.99)
# exit()

best_test_loss = float('inf')
# viz = Visdom()
# viz.line([0.], [0.], win='train_loss', opts=dict(title='train_loss'))
# viz.line([0.], [0.], win='val_loss', opts=dict(title='val_loss'))


for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(model_optimizer, train_data_loader, model, criterion, device)
    eval_loss, perplexity = evaluate(evl_data_loader, model, criterion, device)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print('Epoch:', '%04d' % (epoch + 1),  f'Time: {epoch_mins}m {epoch_secs}s')
    print('\ttrain loss: ', '{:.4f}'.format(train_loss))
    print('\t eval_loss: ', '{:.4f}'.format(eval_loss))
    print('\tperplexity: ', '{:.4f}'.format(perplexity))
    if eval_loss < best_test_loss:
        best_test_loss = eval_loss
        # torch.save(gcn_model.state_dict(), 'save_model/gcn_model.pt')
        torch.save(model.state_dict(), 'save_model/pare.pt')
        # torch.save(trans2_model.state_dict(), 'save_model/multi_loss2.pt')

    # viz.line([train_loss], [epoch], win='train_loss', update='append')
    # viz.line([eval_loss], [epoch], win='val_loss', update='append')
