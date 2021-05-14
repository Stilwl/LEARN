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
    def attention(self,hidden, hidden_prev,t=5):
        # hidden = [2,batch_size,1,dec_num_unit]
        # hidden_prev = [2,batch_size,t,dec_num_unit]
        # energy = [2,batch_size,t,dec_num_unit]
        # attention = [2,batch_size,1,t]
        hidden = hidden.repeat(1, t, 1)
        energy = self.attn(torch.cat((hidden, hidden_prev), dim=2))
        #attention = self.v(energy).transpose(2, 3)
        attention = energy.transpose(1,2)
        return F.softmax(attention, dim=2)
    def forward(self, received):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # received = [batch_size, block_len, n/k]
        # rnn_out = [batch_size,block_len,dec_num_unit]
        rnn_out1, _ = self.dec1_rnns(received)
        rnn_out2, _ = self.dec2_rnns(received)
        copy_out1 =torch.zeros_like(rnn_out1)
        copy_out2 = torch.zeros_like(rnn_out2)
        copy_out1[:, 0:self.args.attn_num, :] = rnn_out1[:, 0:self.args.attn_num, :]
        copy_out2[:,0:self.args.attn_num,:] = rnn_out2[:,0:self.args.attn_num,:]
        for t in range(self.args.attn_num,self.args.block_len):
            a1 = self.attention(rnn_out1[:,t:t+1,:],rnn_out1[:,t-self.args.attn_num:t,:],self.args.attn_num)
            c1 = torch.bmm(a1,rnn_out1[:,t-self.args.attn_num:t,:])
            new_out1 = self.context(torch.cat((c1,rnn_out1[:,t:t+1,:]),dim=2))
            copy_out1[:,t:t+1,:]= new_out1
            a2 = self.attention(rnn_out2[:, t:t + 1, :], rnn_out2[:, t-self.args.attn_num:t, :], self.args.attn_num)
            c2 = torch.bmm(a2,rnn_out2[:,t-self.args.attn_num:t,:])
            new_out2 = self.context(torch.cat((c2, rnn_out2[:, t:t + 1, :]), dim=2))
            copy_out2[:, t:t + 1, :] = new_out2
        for i in range(self.args.block_len):
            if (i>=self.args.block_len-self.args.D-1):
                rt_d = copy_out2[:,self.args.block_len-1:self.args.block_len,:]
            else:
                rt_d = copy_out2[:,i+self.args.D:i+self.args.D+1,:]
            rt = copy_out1[:,i:i+1,:]
            rnn_out = torch.cat((rt, rt_d), dim=2)
            dec_out = self.dec_act(self.dec_outputs(rnn_out))
            if i==0:
                final = dec_out
            else:
                final = torch.cat((final,dec_out),dim=1)
        final = torch.sigmoid(final)

        return final