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

        self.dec1_rnns = RNN_MODEL(args.enc_num_unit+int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                                    num_layers=2, bias=True, batch_first=True,
                                                    dropout=args.dropout)

        self.dec2_rnns = RNN_MODEL(args.enc_num_unit+int(args.code_rate_n/args.code_rate_k),  args.dec_num_unit,
                                        num_layers=2, bias=True, batch_first=True,
                                        dropout=args.dropout)
        #int(args.code_rate_n/args.code_rate_k)
        self.dec_outputs = torch.nn.Linear(2*args.dec_num_unit, 1)
        self.attn = torch.nn.Linear(args.enc_num_unit + args.dec_num_unit, args.dec_num_unit, bias=False)
        self.v = torch.nn.Linear(args.dec_num_unit, 1, bias=False)
    
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
    def attention(self,s, enc_output):
        #enc_output = [batch_size, block_len, enc_num_unit]
        # s = [batch_size, src_len, dec_num_unit]
        s = s.unsqueeze(1).repeat(1, self.args.block_len, 1)
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
    def forward(self, received, s, enc_output_all):
        received = received.type(torch.FloatTensor).to(self.this_device)
        # received = [batch_size, block_len, n/k]
        # s = [batch_size, dec_num_unit]
        for t in range(self.args.block_len):
            # a = [batch_size, 1, block_len]
            # c = [1, batch_size, enc_num_unit]

            if t==0:
                a = self.attention(s, enc_output_all).unsqueeze(1)
                c = torch.bmm(a, enc_output_all)
                rnn_input = torch.cat((received[:, t:t + 1, :], c), dim=2)
                dec_output1, dec_hidden1 = self.dec1_rnns(rnn_input, s.unsqueeze(0).repeat(2, 1, 1))
                rnn_out1 = dec_output1
                dec_output2, dec_hidden2 = self.dec2_rnns(rnn_input, s.unsqueeze(0).repeat(2, 1, 1))
                rnn_out2 = dec_output2
            else:
                a1 = self.attention(s1[-1,:,:].squeeze(0), enc_output_all).unsqueeze(1)
                a2 = self.attention(s2[-1,:,:].squeeze(0), enc_output_all).unsqueeze(1)
                c1 = torch.bmm(a1, enc_output_all)
                c2 = torch.bmm(a2, enc_output_all)
                rnn_input1 = torch.cat((received[:, t:t + 1, :], c1), dim=2)
                rnn_input2 = torch.cat((received[:, t:t + 1, :], c2), dim=2)
                dec_output1, dec_hidden1 = self.dec1_rnns(rnn_input1, s1)
                rnn_out1 = torch.cat((rnn_out1,dec_output1), dim=1)
                dec_output2, dec_hidden2 = self.dec1_rnns(rnn_input2, s2)
                rnn_out2 = torch.cat((rnn_out2, dec_output2), dim=1)
            s1 = dec_hidden1
            s2 = dec_hidden2

        for i in range(self.args.block_len):
            if (i>=self.args.block_len-self.args.D-1):
                rt_d = rnn_out2[:,self.args.block_len-1:self.args.block_len,:]
            else:
                rt_d = rnn_out2[:,i+self.args.D:i+self.args.D+1,:]
            rt = rnn_out1[:,i:i+1,:]
            rnn_out = torch.cat((rt, rt_d), dim=2)
            dec_out = self.dec_outputs(rnn_out)
            if i==0:
                final = dec_out
            else:
                final = torch.cat((final,dec_out),dim=1)
        final = torch.sigmoid(final)

        return final