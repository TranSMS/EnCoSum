import torch
import torch.nn as nn


class Seq2SeqEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, dropout=0.5, source_vocab_size=1000):
        super(Seq2SeqEncoder, self).__init__()

        self.gru_layer = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self.embedding_table = nn.Embedding(source_vocab_size, embedding_dim)
        self.hidden2label = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input_embedding = self.embedding_table(input)
        # input_embedding =
        output_state, final_hidden = self.gru_layer(input_embedding)
        output_state = self.dropout(output_state)
        output = self.hidden2label(output_state)
        return output
