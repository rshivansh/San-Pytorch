import torch
import torch.nn as nn
from torch.autograd import Variable


class Attention(nn.Module): # Extend PyTorch's Module class
    def __init__(self, input_size, att_size, img_seq_size, output_size, drop_ratio):
        super(Attention, self).__init__() # Must call super __init__()
        self.input_size = input_size
        self.att_size = att_size
        self.img_seq_size = img_seq_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio

        self.tan = nn.Tanh()
        self.dp = nn.Dropout(drop_ratio)
        self.sf = nn.Softmax()

        self.fc11 = nn.Linear(input_size, 768, bias=True)
        self.fc111 = nn.Linear(768, 640, bias=True)
        self.fc12 = nn.Linear(input_size, 768, bias=False)
        self.fc121 = nn.Linear(768, 640, bias=False)
        self.linear_second = nn.Linear(640, att_size, bias=False)
        self.fc13 = nn.Linear(att_size, 1, bias=True)

        self.fc21 = nn.Linear(input_size, att_size, bias=True)
        self.fc22 = nn.Linear(input_size, att_size, bias=False)
        self.fc23 = nn.Linear(att_size, 1, bias=True)

        self.fc = nn.Linear(input_size, output_size, bias=True)

        # d = input_size | m = img_seq_size | k = att_size
    def forward(self, ques_feat, img_feat):  # ques_feat -- [batch, d] | img_feat -- [batch_size, m, d]
        #  print(img_feat.size(), ques_feat.size())
        B = ques_feat.size(0)

        # Stack 1
          x = F.tanh(self.linear_first(outputs))       
        x = self.linear_second(x)       
        x = self.softmax(x,1) 
        ques_emb_1 = self.fc11(ques_feat) 
        ques_emb_1 = self.fc111(ques_emb_1) # [batch_size, att_size]
        img_emb_1 = self.fc12(img_feat)
        img_emb_1 = self.fc121(img_emb_1)

        h1 = self.tan(ques_emb_1.view(B, 1, self.att_size) + img_emb_1)
        h1_emb = self.linear_second(h1) 
        
        p1 = self.sf(h1_emb.view(-1, self.img_seq_size)).view(B, 1, self.img_seq_size)

        # Weighted sum
        img_att1 = p1.matmul(img_feat)
        u1 = ques_feat + img_att1.view(-1, self.input_size)

        # Stack 2
        ques_emb_2 = self.fc21(u1)  # [batch_size, att_size]
        img_emb_2 = self.fc22(img_feat)

        h2 = self.tan(ques_emb_2.view(B, 1, self.att_size) + img_emb_2)

        h2_emb = self.fc23(self.dp(h2))
        p2 = self.sf(h2_emb.view(-1, self.img_seq_size)).view(B, 1, self.img_seq_size)

        # Weighted sum
        img_att2 = p2.matmul(img_feat)
        u2 = u1 + img_att2.view(-1, self.input_size)

        # score
        score = self.fc(u2)

    return score
