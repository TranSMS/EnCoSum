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
train_num = 69708 
max_ast_node = 80  
src_max_length = 300  
md_max_len = 6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=150, help="number of epochs")
parser.add_argument("-sn", "--max_ast_node_num", type=int, default=100, help="AST graph max node number")
parser.add_argument("-s", "--src_max_len", type=int, default=300, help="maximum sequence len")
parser.add_argument("-n", "--nl_max_len", type=int, default=30, help="maximum nl len")
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

parser.add_argument("-tn", "--train_num", type=int, default=69708, help="training number")

args = parser.parse_args()

tgt_vocab_size, tgt_inv_vocab_dict, dec_inputs, tgt_vocab, dec_outputs = load_nl_data('dataset/example_nl.txt', nl_max_len=args.nl_max_len)
src_vocab_size, enc_inputs, md_inputs, src_vocab = load_code_data('dataset/example_code.txt', 'dataset/example_method.txt', args.src_max_len, args.md_max_len)
# print(src_vocab)
# print(tgt_vocab)
# print(tgt_vocab_size)
# print(md_inputs)
# exit()
A, A2, A3, A4, A5 = read_batchA('dataset/example_ast.txt', args.max_ast_node_num)
X = get_embed('dataset/example_ast.txt', args.max_ast_node_num)

A_train = A[0:args.train_num]
A_test = A[args.train_num:len(A)]
# print(A_2)
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

# exit()
# dataset = MySet(A, X, A2, A3, A4, A5, enc_inputs, dec_inputs, dec_outputs)
train_data = MySet(A_train, X_train, A2_train, A3_train, A4_train, A5_train, enc_train, md_train, dec_in_train, dec_out_train)
evl_data = MySet(A_test, X_test, A2_test, A3_test, A4_test, A5_test, enc_test, md_test, dec_in_test, dec_out_test)
# train_data, evl_data = random_split(dataset, [1040, 260])
# exit()
my_sampler1 = MySampler(train_data, args.batch_size)
my_sampler2 = MySampler(evl_data, args.batch_size)
evl_data_loader = DataLoader(evl_data, batch_sampler=my_sampler2)
train_data_loader = DataLoader(train_data, batch_sampler=my_sampler1)

# trans_loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=batch_size, shuffle=True)
# gcn_model = GCNEncoder().to(device)
model = Model(src_vocab_size, tgt_vocab_size, max_ast_node=args.max_ast_node_num, src_max_length=args.src_max_len, method_max_length=args.md_max_len,
                          nfeat=args.nfeat_dim, nhid=args.nhid_dim, nout=args.nout_dim, d_model=args.dmodel, batch_size=args.batch_size, dropout=args.dropout,
                          d_k=args.dk, d_v=args.dv, d_ff=args.dff, n_heads=args.attn_heads, n_layers=args.layers, device=device).to(device)
# trans2_model = Transformer2(src_vocab_size, tgt_vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
LEARNING_RATE = args.lr
N_EPOCHS = args.epochs
# gcn_optimizer = optim.SGD(gcn_model.parameters(), lr=0.0001, momentum=0.99)
model_optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.99)

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
        torch.save(model.state_dict(), 'save_model/pare.pt')
       
def beam_search(trans_model, enc_input, ast_outputs, ast_embed, start_symbol):  # 变动

    enc_outputs, enc_self_attns = trans_model.encoder(enc_input)
    dec_input = torch.zeros(1, nl_max_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, nl_max_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _, _ = trans_model.decoder1(dec_input, enc_input, enc_outputs, ast_outputs, ast_embed)  # 变动
        projected = trans_model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


def nltk_sentence_bleu(hypotheses, references, order=4):
    refs = []
    count = 0
    total_score = 0.0
    cc = SmoothingFunction()
    for hyp, ref in zip(hypotheses, references):
        hyp = hyp.split()
        ref = ref.split()
        refs.append([ref])
        if len(hyp) < order:
            continue
        else:
            score = nltk.translate.bleu([ref], hyp, smoothing_function=cc.method4)
            total_score += score
            count += 1
    avg_score = total_score / count
    hpy2 = []
    for i in hypotheses:
        # print(i)
        i = i.split()
        hpy2.append(i)
    return avg_score


def meteor_score1(hypothesis, reference):
    count = 0
    total_score = 0.0
    for i in range(len(hypothesis)):
        score = round(meteor_score([reference[i]], hypothesis[i]), 4)
        # print(score)
        # exit()
        total_score += score
        count += 1
    avg_score = total_score/count
    # print('METEOR_score: %.4f' % avg_score)
    return avg_score


def predict():

    model.load_state_dict(torch.load('save_model/trans_loss2.pt'))
    model.eval()
    start_time = time.time()
    a, x, a2, a3, a4, a5, inputs, _, _ = next(iter(evl_data_loader))
    q = []
    for j in range(len(inputs)):
        a, x, a2, a3, a4, a5 = a.to(device), x.to(device), a2.to(device), a3.to(device), a4.to(device), a5.to(device)

        ast_outputs, ast_embed = model.gcn_encoder(x[j].unsqueeze(0), a[j].unsqueeze(0), a2[j].unsqueeze(0), a3[j].unsqueeze(0), a4[j].unsqueeze(0), a5[j].unsqueeze(0))
        greedy_dec_input = beam_search(model, inputs[j].view(1, -1).to(device), ast_outputs, ast_embed, start_symbol=tgt_vocab['SOS'])  # 变动
        pred, _, _, _, _ = model(inputs[j].view(1, -1).to(device), greedy_dec_input, x[j].unsqueeze(0), a[j].unsqueeze(0), a2[j].unsqueeze(0), a3[j].unsqueeze(0), a4[j].unsqueeze(0), a5[j].unsqueeze(0))  # 变动
        pred = pred.data.max(1, keepdim=True)[1]
        for i in range(len(pred)):
            if i > 0 and pred[i] == 3:
                pred = pred[0:i+1]
                break
            else:
                continue
        x1 = [tgt_inv_vocab_dict[n.item()] for n in pred.squeeze()]
        q.append(x1)
    # print(q)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Time: {epoch_mins}m {epoch_secs}s')
    pred1 = []
    for k in q:
        s = " ".join(k)
        pred1.append(s)
    # print(pred1)
    with open('data/hyp.txt', 'w', encoding='utf-8') as ff:
        for z in pred1:
            ff.writelines(z + '\n')
    ref = []
    with open('data/ref2.txt', 'r', encoding='utf-8') as f:  # ref1
        lines = f.readlines()

        for line in lines:
            line = line.strip('\n')
            ref.append(line)

    avg_score = nltk_sentence_bleu(pred1, ref)
    meteor = meteor_score1(pred1, ref)
    print('S_BLEU: %.4f' % avg_score)
    # print('C-BLEU: %.4f' % corup_BLEU)
    print('METEOR: %.4f' % meteor)
    rouge = Rouge()
    rough_score = rouge.get_scores(pred1, ref, avg=True)
    print(' ROUGE: ', rough_score)

if __name__ == '__main__':
    predict()
