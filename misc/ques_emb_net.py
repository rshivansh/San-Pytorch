import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class QuestionEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, rnn_size, num_layers, dropout, seq_length, use_gpu):
        super(QuestionEmbedding, self).__init__() # Must call super __init__()

	self.use_gpu = use_gpu
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.lookuptable = nn.Linear(vocab_size, emb_size, bias=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                num_layers=num_layers, bias=True,
                batch_first=True, dropout=dropout)

        return

    def forward(self, ques_vec, ques_len):            # forward(self, ques_vec, ques_len) | ques_vec: [batch_size, 26]
        B, W = ques_vec.size()

        # Add 1 to vocab_size, since word idx from 0 to vocab_size inclusive
        one_hot_vec = torch.zeros(B, W, self.vocab_size+1).scatter_(2,
                        ques_vec.data.type('torch.LongTensor').view(B, W, 1), 1)

        # To remove additional column in one_hot, use slicing
        one_hot_vec = Variable(one_hot_vec[:,:,1:], requires_grad=False)
        if self.use_gpu and torch.cuda.is_available():
            one_hot_vec = one_hot_vec.cuda()

        x = self.lookuptable(one_hot_vec)

        # emb_vec: [batch_size or B, 26 or W, emb_size]
        emb_vec = self.dropout(self.tanh(x))

        # h: [batch_size or B, 26 or W, hidden_size]
        h, _ = self.LSTM(emb_vec)

        x = torch.LongTensor(ques_len - 1)
        mask = torch.zeros(B, W).scatter_(1, x.view(-1, 1), 1)
        mask = Variable(mask.view(B, W, 1), requires_grad=False)
        if self.use_gpu and torch.cuda.is_available():
            mask = mask.cuda()

        h = h.transpose(1,2)
        # print(h.size(), mask.size())

        # output: [B, hidden_size]
    return torch.bmm(h, mask).view(B, -1)
