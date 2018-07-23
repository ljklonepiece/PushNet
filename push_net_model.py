import torch # arrays on GPU
import torch.autograd as autograd #build a computational graph
import torch.nn as nn ## neural net library
import torch.nn.functional as F ## most non-linearities are here
import torch.optim as optim # optimization package

import torchvision.models as models
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config as args

WIDTH = 128
HEIGHT = 106

ACT_SIZE = 4
ACT_FET = 20
LSTM_IN_SIZE = 80
HIDDEN_SIZE = 80
COM_OUT = 2
SIM_SIZE = 3

chan_layer_1 = 16
chan_layer_2 = 16
chan_layer_3 = 32
chan_layer_4 = 32
pool_size = 3

''' CNN Module '''
class COM_CNN(nn.Module):
    def __init__(self):
        super(COM_CNN, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=1,
                    out_channels=chan_layer_1,
                    kernel_size=3,
                    stride=1,
                    padding=2,
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(chan_layer_1, chan_layer_2, 3, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(chan_layer_2, chan_layer_3, 3, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(chan_layer_3, chan_layer_4, 3, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_size),
                )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight, gain=np.sqrt(2.0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return x

''' Push Net '''
class COM_net_sim(nn.Module):
    def __init__(self, batch_size):
        super(COM_net_sim, self).__init__()
        self.cnn = COM_CNN()
        f_len = self.get_img_feature_len()
        self.batch_size = batch_size
        self.linear_act = nn.Linear(ACT_SIZE, ACT_FET)
        self.linear_act_img_curr = nn.Linear(f_len + ACT_FET,  LSTM_IN_SIZE)

        self.linear_img_img = nn.Linear(f_len, SIM_SIZE)
        self.lstm = nn.LSTM(LSTM_IN_SIZE, HIDDEN_SIZE, 1, batch_first=True)

        self.linear_fnext_out = nn.Linear(HIDDEN_SIZE, f_len)
        self.linear_com_out = nn.Linear(HIDDEN_SIZE, COM_OUT)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        ### two variables: (h0, c0)
        return (autograd.Variable(torch.zeros(1, self.batch_size,
            HIDDEN_SIZE).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size,
                    HIDDEN_SIZE).cuda()))

    def get_img_feature_len(self):
        test = torch.rand(1, 1, WIDTH, HEIGHT)
        return self.cnn(Variable(test)).size()[1]

    def forward(self, a0, I1, a1, Ig, lengths, bs):
        ''' get image and action feature representation'''

        ''' flatten mini-batch sequence of images'''
        f1 = self.cnn(I1.view(-1, 1, 106, 128))
        fg = self.cnn(Ig.view(-1, 1, 106, 128))
        ''' flatten mini-batch sequence of action'''
        fa1 = self.linear_act(a1.view(-1, 4))


        ''' combine img and previous action feature to form one-step history'''
        cat_f1_fa1 = torch.cat((f1, fa1), 1)
        lstm_inp = self.linear_act_img_curr(cat_f1_fa1)

        ''' pack sequence to feed LSTM '''
        lstm_inp = pack_padded_sequence(lstm_inp.view(bs, -1, LSTM_IN_SIZE),
                lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)

        ''' unpack sequence to feed in linear layer'''
        lstm_out_unpad, lengths_new = pad_packed_sequence(lstm_out, batch_first=True)

        com_out = self.linear_com_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        fnext = self.linear_fnext_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))

        ''' evaluate similarity between target internal state & true internal state'''
        output = self.linear_img_img(torch.abs(fnext - fg))

        ''' squash value between (0, 1)'''

        sim = F.sigmoid(output)
        com_out = F.sigmoid(com_out)

        ''' pack into sequence for output'''
        sim_pack = pack_padded_sequence(sim.view(bs, -1, SIM_SIZE), lengths, batch_first=True)
        com_pack = pack_padded_sequence(com_out.view(bs, -1, 2), lengths, batch_first=True)

        return sim_pack, com_pack

''' Push-Net-sim '''
class COM_net_sim_only(nn.Module):
    def __init__(self, batch_size):
        super(COM_net_sim_only, self).__init__()
        self.cnn = COM_CNN()
        f_len = self.get_img_feature_len()
        self.batch_size = batch_size
        self.linear_act = nn.Linear(ACT_SIZE, ACT_FET)
        self.linear_act_img_curr = nn.Linear(f_len + ACT_FET,  LSTM_IN_SIZE)

        self.linear_img_img = nn.Linear(f_len, SIM_SIZE)
        self.lstm = nn.LSTM(LSTM_IN_SIZE, HIDDEN_SIZE, 1, batch_first=True)

        self.linear_fnext_out = nn.Linear(HIDDEN_SIZE, f_len)
        self.linear_com_out = nn.Linear(HIDDEN_SIZE, COM_OUT)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        ### two variables: (h0, c0)
        return (autograd.Variable(torch.zeros(1, self.batch_size,
            HIDDEN_SIZE).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size,
                    HIDDEN_SIZE).cuda()))

    def get_img_feature_len(self):
        test = torch.rand(1, 1, WIDTH, HEIGHT)
        return self.cnn(Variable(test)).size()[1]

    def forward(self, a0, I1, a1, Ig, lengths, bs):
        ''' get image and action feature representation'''

        ''' flatten mini-batch sequence of images'''
        f1 = self.cnn(I1.view(-1, 1, 106, 128))
        fg = self.cnn(Ig.view(-1, 1, 106, 128))
        ''' flatten mini-batch sequence of action'''
        fa1 = self.linear_act(a1.view(-1, 4))

        ''' combine img and previous action feature to form one-step history'''
        cat_f1_fa1 = torch.cat((f1, fa1), 1)
        lstm_inp = self.linear_act_img_curr(cat_f1_fa1)

        ''' pack sequence to feed LSTM '''
        lstm_inp = pack_padded_sequence(lstm_inp.view(bs, -1, LSTM_IN_SIZE),
                lengths, batch_first=True)
        lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)

        ''' unpack sequence to feed in linear layer'''
        lstm_out_unpad, lengths_new = pad_packed_sequence(lstm_out, batch_first=True)

        com_out = self.linear_com_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        fnext = self.linear_fnext_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))


        ''' evaluate similarity between target internal state & true internal state'''
        output = self.linear_img_img(torch.abs(fnext - fg))

        ''' squash value between (0, 1)'''

        sim = F.sigmoid(output)

        ''' pack into sequence for output'''
        sim_pack = pack_padded_sequence(sim.view(bs, -1, SIM_SIZE), lengths, batch_first=True)

        return sim_pack

''' Push-Net-nomem '''
class COM_net_nomem(nn.Module):
    def __init__(self, batch_size):
        super(COM_net_nomem, self).__init__()
        self.cnn = COM_CNN()
        f_len = self.get_img_feature_len()
        self.batch_size = batch_size
        self.linear_act = nn.Linear(ACT_SIZE, ACT_FET)
        self.linear_act_img_curr = nn.Linear(f_len + ACT_FET,  f_len)

        self.linear_img_img = nn.Linear(f_len, SIM_SIZE)


    def init_hidden(self):
        ### two variables: (h0, c0)
        return (autograd.Variable(torch.zeros(1, self.batch_size,
            HIDDEN_SIZE).cuda()),
                autograd.Variable(torch.zeros(1, self.batch_size,
                    HIDDEN_SIZE).cuda()))

    def get_img_feature_len(self):
        test = torch.rand(1, 1, WIDTH, HEIGHT)
        return self.cnn(Variable(test)).size()[1]

    def forward(self, a0, I1, a1, Ig, lengths, bs):
        ''' get image and action feature representation'''

        ''' flatten mini-batch sequence of images'''
        f1 = self.cnn(I1.view(-1, 1, 106, 128))
        fg = self.cnn(Ig.view(-1, 1, 106, 128))
        ''' flatten mini-batch sequence of action'''
        fa1 = self.linear_act(a1.view(-1, 4))

        ''' combine img and previous action feature to form one-step history'''
        cat_f1_fa1 = torch.cat((f1, fa1), 1)
        fnext = self.linear_act_img_curr(cat_f1_fa1)

        ''' evaluate similarity between target internal state & true internal state'''
        output = self.linear_img_img(torch.abs(fnext - fg))

        ''' squash value between (0, 1)'''

        sim = F.sigmoid(output)
        ''' pack into sequence for output'''
        sim_pack = pack_padded_sequence(sim.view(bs, -1, SIM_SIZE), lengths, batch_first=True)

        return sim_pack

def get_num_parameters(model):
    model_parm = filter(lambda p:p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parm])

if __name__=='__main__':
    net = COM_net_sim(args.batch_size)
    print get_num_parameters(net)





