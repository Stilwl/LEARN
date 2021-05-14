import torch.nn.functional as F
from torch.nn import Parameter
import torch

from numpy import arange
from numpy.random import mtrand
import numpy as np

class DEC(torch.nn.Module):
    def __init__(self, args):
        super(DEC, self).__init__()
        self.args = args

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.this_device = torch.device("cuda" if use_cuda else "cpu")

        if args.dec_rnn == 'gru':
            RNN_MODEL = torch.nn.GRU
        elif args.dec_rnn == 'lstm':
            RNN_MODEL = torch.nn.LSTM
        else:
            RNN_MODEL = torch.nn.RNN

        self.dropout = torch.nn.Dropout(args.dropout)

        self.dec1_rnns = RNN_MODEL(int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                                    num_layers=2, bias=True, batch_first=True,
                                                    dropout=args.dropout)

        self.dec2_rnns = RNN_MODEL(int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                        num_layers=2, bias=True, batch_first=True,
                                        dropout=args.dropout)
        #int(args.code_rate_n/args.code_rate_k)
        self.dec_outputs = torch.nn.Linear(2*args.dec_num_unit, 1)
        self.attn = torch.nn.Linear(2*args.dec_num_unit, 1)
        #self.v = torch.nn.Linear(args.dec_num_unit, 1, bias=False)
        self.context = torch.nn.Linear(2*args.dec_num_unit, args.dec_num_unit)
    
    def dec_act(self, inputs):
        if self.args.dec_act == 'tanh':
            return  F.tanh(inputs)
        elif self.args.dec_act == 'elu':
            return F.elu(inputs)
        elif self.args.dec_act == 'relu':
            return F.relu(inputs)
        elif self.args.dec_act == 'selu':
            return F.selu(inputs)
        elif self.args.dec_act == 'sigmoid':
            return F.sigmoid(inputs)
        elif self.args.dec_act == 'linear':
            return inputs
        else:
            return inputs
    def attention(self,hidden, hidden_prev,t):
        # hidden = [2,batch_size,1,dec_num_unit]
        # hidden_prev = [2,batch_size,t,dec_num_unit]
        # energy = [2,batch_size,t,dec_num_unit]
        # attention = [2,batch_size,1,t]
        hidden = hidden.unsqueeze(2).repeat(1, 1, t, 1)
        energy = self.attn(torch.cat((hidden, hidden_prev), dim=3))
        #attention = self.v(energy).transpose(2, 3)
        attention = energy.transpose(2,3)
        return F.softmax(attention, dim=3)
    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # received = [batch_size, block_len, n/k]
        for t in range(self.args.block_len):
            # a = [2, batch_size, 1, t]
            # c = [2, batch_size, 1, dec_num_unit]
            # hiddens = [2,batch_size,t->t+1,dec_num_unit]
            # hidden1 = [2,batch_size,1,dec_num_unit]
            # dec_out1 = [batch_size,1,dec-num_unit]
            if t == 0:
                dec_out1, hidden1 = self.dec1_rnns(received[:, t:t + 1, :])
                dec_out2, hidden2 = self.dec2_rnns(received[:, t:t + 1, :])
                rnn_out1 = dec_out1
                rnn_out2 = dec_out2
                hiddens1 = hidden1.unsqueeze(2)
                hiddens2 = hidden2.unsqueeze(2)
            else:
                dec_out1, hidden1 = self.dec1_rnns(received[:, t:t + 1, :],hidden1)
                dec_out2, hidden2 = self.dec2_rnns(received[:, t:t + 1, :], hidden2)
                rnn_out1 = torch.cat((rnn_out1, dec_out1), dim=1)
                rnn_out2 = torch.cat((rnn_out2, dec_out2), dim=1)
                a1 = self.attention(hidden1,hiddens1,t)
                a2 = self.attention(hidden2, hiddens2, t)
                #c1 = torch.cat((torch.bmm(a1[0].squeeze(0), hiddens1[0].squeeze(0)).unsqueeze(0),torch.bmm(a1[1].squeeze(0), hiddens1[1].squeeze(0)).unsqueeze(0)),dim=0)
                c1 = torch.matmul(a1,hiddens1)
                #c2 = torch.cat((torch.bmm(a2[0].squeeze(0), hiddens2[0].squeeze(0)).unsqueeze(0),torch.bmm(a2[1].squeeze(0), hiddens2[1].squeeze(0)).unsqueeze(0)),dim=0)
                c2 = torch.matmul(a2, hiddens2)
                hiddens1 = torch.cat((hiddens1, hidden1.unsqueeze(2)), dim=2)
                hiddens2 = torch.cat((hiddens2, hidden2.unsqueeze(2)), dim=2)
                hidden1 = self.context(torch.cat((c1.squeeze(2),hidden1),dim=2))
                hidden2 = self.context(torch.cat((c2.squeeze(2), hidden2), dim=2))

        for i in range(self.args.block_len):
            if (i>=self.args.block_len-self.args.D-1):
                rt_d = rnn_out2[:,self.args.block_len-1:self.args.block_len,:]
            else:
                rt_d = rnn_out2[:,i+self.args.D:i+self.args.D+1,:]
            rt = rnn_out1[:,i:i+1,:]
            rnn_out = torch.cat((rt, rt_d), dim=2)
            dec_out = self.dec_act(self.dec_outputs(rnn_out))
            if i==0:
                final = dec_out
            else:
                final = torch.cat((final,dec_out),dim=1)
        final = torch.sigmoid(final)

        return final