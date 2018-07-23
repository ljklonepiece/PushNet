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
#LSTM_IN_SIZE = 60
LSTM_IN_SIZE = 80
HIDDEN_SIZE = 80
#LSTM_OUT_SIZE = 25
COM_OUT = 2
SIM_SIZE = 3

#chan_layer_1 = 8
chan_layer_1 = 16
chan_layer_2 = 16
chan_layer_3 = 32
#chan_layer_4 = 16
chan_layer_4 = 32
pool_size = 3

#ACT_FET = 10
#LSTM_IN_SIZE = 15
#HIDDEN_SIZE = 20
#LSTM_OUT_SIZE = 25
#COM_OUT = 2
#SIM_SIZE = 3
#
#chan_layer_1 = 8
#chan_layer_2 = 16
#chan_layer_3 = 32
#pool_size = 3

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
                #init.constant(m.bias, 0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return x

class COM_net(nn.Module):
    def __init__(self, batch_size):
        super(COM_net, self).__init__()
        self.cnn = COM_CNN()
        f_len = self.get_img_feature_len()
        self.batch_size = batch_size
        #print f_len
        self.linear_act = nn.Linear(ACT_SIZE, ACT_FET)
        self.linear_act_prev_img = nn.Linear(f_len + ACT_FET,  LSTM_IN_SIZE)
        self.linear_act_curr_img = nn.Linear(f_len + ACT_FET + LSTM_OUT_SIZE, f_len)
        self.linear_img_img = nn.Linear(f_len, SIM_SIZE)
        self.lstm = nn.LSTM(LSTM_IN_SIZE, HIDDEN_SIZE, 1, batch_first=True)
        self.linear_lstm_out = nn.Linear(HIDDEN_SIZE, LSTM_OUT_SIZE)
        self.linear_com_out = nn.Linear(LSTM_OUT_SIZE, COM_OUT)
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

        ''' 1. flatten mini-batch sequence of images'''
        f1 = self.cnn(I1.view(-1, 1, 106, 128))
        fg = self.cnn(Ig.view(-1, 1, 106, 128))
        ''' 2. flatten mini-batch sequence of action'''
        fa0 = self.linear_act(a0.view(-1, 4))
        fa1 = self.linear_act(a1.view(-1, 4))

        #print f1.size() ## bs*seq x 8160
        #print fa0.size() ## bs*seq x 10

        ''' combine img and previous action feature to form one-step history'''
        cat_f1_fa0 = torch.cat((f1, fa0), 1)
        #print cat_f1_fa0.size()
        lstm_inp = self.linear_act_prev_img(cat_f1_fa0)
        #print lstm_inp.size() ## 200 x 15

        ''' pack sequence to feed LSTM '''
        #lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)
        lstm_inp = pack_padded_sequence(lstm_inp.view(bs, -1, LSTM_IN_SIZE),
                lengths, batch_first=True)
        #print lstm_inp.data.size() ## 188 x 15
        lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)
        #lstm_out, self.hidden = self.lstm(lstm_inp)

        ''' unpack sequence to feed in linear layer'''
        #print lstm_out.data.size() ## 188 x hidden_size
        lstm_out_unpad, lengths_new = pad_packed_sequence(lstm_out, batch_first=True)
        #print lstm_out_unpad.size() ## bs x seq x hidden_size

        lstm_out = self.linear_lstm_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        com_out = self.linear_com_out(lstm_out)
        #print com_out.size() ## bs*seq x 2

        ''' combine img and current action to form one-step transition'''
        combined = torch.cat([f1, fa1], 1)

        ''' combine transition with current estimate of internal state'''
        combined = torch.cat([combined, lstm_out], 1)
        ''' get next internal state'''
        fnext = self.linear_act_curr_img(combined)

        ''' evaluate similarity between target internal state & true internal state'''
        output = self.linear_img_img(torch.abs(fnext - fg))

        ''' squash value between (0, 1)'''

        sim = F.sigmoid(output)
        com_out = F.sigmoid(com_out)

        #print sim.size() ## bs*seq x SIM_SIZE
        #print com_out.size() ## bs*seq x 2
        ''' pack into sequence for output'''
        sim_pack = pack_padded_sequence(sim.view(bs, -1, SIM_SIZE), lengths, batch_first=True)
        com_pack = pack_padded_sequence(com_out.view(bs, -1, 2), lengths, batch_first=True)

        #print sim_pack.data.size() ## 188 x SIM_SIZE
        #print com_pack.data.size() ## 188 x 2

        return sim_pack, com_pack

class COM_net_sim(nn.Module):
    def __init__(self, batch_size):
        super(COM_net_sim, self).__init__()
        self.cnn = COM_CNN()
        f_len = self.get_img_feature_len()
        self.batch_size = batch_size
        #print f_len
        self.linear_act = nn.Linear(ACT_SIZE, ACT_FET)
        self.linear_act_img_curr = nn.Linear(f_len + ACT_FET,  LSTM_IN_SIZE)
        #self.linear_act_curr_img = nn.Linear(f_len + ACT_FET + LSTM_OUT_SIZE, f_len)

        self.linear_img_img = nn.Linear(f_len, SIM_SIZE)
        self.lstm = nn.LSTM(LSTM_IN_SIZE, HIDDEN_SIZE, 1, batch_first=True)
        #self.linear_lstm_out = nn.Linear(HIDDEN_SIZE, LSTM_OUT_SIZE)

        #self.linear_fnext_out = nn.Linear(LSTM_OUT_SIZE, f_len)
        #self.linear_com_out = nn.Linear(LSTM_OUT_SIZE, COM_OUT)

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

        ''' 1. flatten mini-batch sequence of images'''
        f1 = self.cnn(I1.view(-1, 1, 106, 128))
        fg = self.cnn(Ig.view(-1, 1, 106, 128))
        ''' 2. flatten mini-batch sequence of action'''
        #fa0 = self.linear_act(a0.view(-1, 4))
        fa1 = self.linear_act(a1.view(-1, 4))

        #print f1.size() ## bs*seq x 8160
        #print fa0.size() ## bs*seq x 10

        ''' combine img and previous action feature to form one-step history'''
        #cat_f1_fa0 = torch.cat((f1, fa0), 1)
        cat_f1_fa1 = torch.cat((f1, fa1), 1)
        #print cat_f1_fa0.size()
        #lstm_inp = self.linear_act_prev_img(cat_f1_fa0)
        lstm_inp = self.linear_act_img_curr(cat_f1_fa1)
        #print lstm_inp.size() ## 200 x 15

        ''' pack sequence to feed LSTM '''
        #lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)
        lstm_inp = pack_padded_sequence(lstm_inp.view(bs, -1, LSTM_IN_SIZE),
                lengths, batch_first=True)
        #print lstm_inp.data.size() ## 188 x 15
        lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)
        #lstm_out, self.hidden = self.lstm(lstm_inp)

        ''' unpack sequence to feed in linear layer'''
        #print lstm_out.data.size() ## 188 x hidden_size
        lstm_out_unpad, lengths_new = pad_packed_sequence(lstm_out, batch_first=True)
        #print lstm_out_unpad.size() ## bs x seq x hidden_size

        #lstm_out = self.linear_lstm_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        com_out = self.linear_com_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        fnext = self.linear_fnext_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        #com_out = self.linear_com_out(lstm_out)
        #print com_out.size() ## bs*seq x 2

        ''' combine img and current action to form one-step transition'''
        #combined = torch.cat([f1, fa1], 1)

        ''' combine transition with current estimate of internal state'''
        #combined = torch.cat([combined, lstm_out], 1)
        ''' get next internal state'''
        #fnext = self.linear_act_curr_img(combined)

        ''' evaluate similarity between target internal state & true internal state'''
        output = self.linear_img_img(torch.abs(fnext - fg))

        ''' squash value between (0, 1)'''

        sim = F.sigmoid(output)
        com_out = F.sigmoid(com_out)

        #print sim.size() ## bs*seq x SIM_SIZE
        #print com_out.size() ## bs*seq x 2
        ''' pack into sequence for output'''
        sim_pack = pack_padded_sequence(sim.view(bs, -1, SIM_SIZE), lengths, batch_first=True)
        com_pack = pack_padded_sequence(com_out.view(bs, -1, 2), lengths, batch_first=True)

        #print sim_pack.data.size() ## 188 x SIM_SIZE
        #print com_pack.data.size() ## 188 x 2

        return sim_pack, com_pack

class COM_net_sim_only(nn.Module):
    def __init__(self, batch_size):
        super(COM_net_sim_only, self).__init__()
        self.cnn = COM_CNN()
        f_len = self.get_img_feature_len()
        self.batch_size = batch_size
        #print f_len
        self.linear_act = nn.Linear(ACT_SIZE, ACT_FET)
        self.linear_act_img_curr = nn.Linear(f_len + ACT_FET,  LSTM_IN_SIZE)
        #self.linear_act_curr_img = nn.Linear(f_len + ACT_FET + LSTM_OUT_SIZE, f_len)

        self.linear_img_img = nn.Linear(f_len, SIM_SIZE)
        self.lstm = nn.LSTM(LSTM_IN_SIZE, HIDDEN_SIZE, 1, batch_first=True)
        #self.linear_lstm_out = nn.Linear(HIDDEN_SIZE, LSTM_OUT_SIZE)

        #self.linear_fnext_out = nn.Linear(LSTM_OUT_SIZE, f_len)
        #self.linear_com_out = nn.Linear(LSTM_OUT_SIZE, COM_OUT)

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

        ''' 1. flatten mini-batch sequence of images'''
        f1 = self.cnn(I1.view(-1, 1, 106, 128))
        fg = self.cnn(Ig.view(-1, 1, 106, 128))
        ''' 2. flatten mini-batch sequence of action'''
        #fa0 = self.linear_act(a0.view(-1, 4))
        fa1 = self.linear_act(a1.view(-1, 4))

        #print f1.size() ## bs*seq x 8160
        #print fa0.size() ## bs*seq x 10

        ''' combine img and previous action feature to form one-step history'''
        #cat_f1_fa0 = torch.cat((f1, fa0), 1)
        cat_f1_fa1 = torch.cat((f1, fa1), 1)
        #print cat_f1_fa0.size()
        #lstm_inp = self.linear_act_prev_img(cat_f1_fa0)
        lstm_inp = self.linear_act_img_curr(cat_f1_fa1)
        #print lstm_inp.size() ## 200 x 15

        ''' pack sequence to feed LSTM '''
        #lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)
        lstm_inp = pack_padded_sequence(lstm_inp.view(bs, -1, LSTM_IN_SIZE),
                lengths, batch_first=True)
        #print lstm_inp.data.size() ## 188 x 15
        lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)
        #lstm_out, self.hidden = self.lstm(lstm_inp)

        ''' unpack sequence to feed in linear layer'''
        #print lstm_out.data.size() ## 188 x hidden_size
        lstm_out_unpad, lengths_new = pad_packed_sequence(lstm_out, batch_first=True)
        #print lstm_out_unpad.size() ## bs x seq x hidden_size

        #lstm_out = self.linear_lstm_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        com_out = self.linear_com_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        fnext = self.linear_fnext_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        #com_out = self.linear_com_out(lstm_out)
        #print com_out.size() ## bs*seq x 2

        ''' combine img and current action to form one-step transition'''
        #combined = torch.cat([f1, fa1], 1)

        ''' combine transition with current estimate of internal state'''
        #combined = torch.cat([combined, lstm_out], 1)
        ''' get next internal state'''
        #fnext = self.linear_act_curr_img(combined)

        ''' evaluate similarity between target internal state & true internal state'''
        output = self.linear_img_img(torch.abs(fnext - fg))

        ''' squash value between (0, 1)'''

        sim = F.sigmoid(output)
        #com_out = F.sigmoid(com_out)

        #print sim.size() ## bs*seq x SIM_SIZE
        #print com_out.size() ## bs*seq x 2
        ''' pack into sequence for output'''
        sim_pack = pack_padded_sequence(sim.view(bs, -1, SIM_SIZE), lengths, batch_first=True)
        #com_pack = pack_padded_sequence(com_out.view(bs, -1, 2), lengths, batch_first=True)

        #print sim_pack.data.size() ## 188 x SIM_SIZE
        #print com_pack.data.size() ## 188 x 2

        #return sim_pack, com_pack
        return sim_pack

class COM_net_nomem(nn.Module):
    def __init__(self, batch_size):
        super(COM_net_nomem, self).__init__()
        self.cnn = COM_CNN()
        f_len = self.get_img_feature_len()
        self.batch_size = batch_size
        #print f_len
        self.linear_act = nn.Linear(ACT_SIZE, ACT_FET)
        #self.linear_act_img_curr = nn.Linear(f_len + ACT_FET,  LSTM_IN_SIZE)
        self.linear_act_img_curr = nn.Linear(f_len + ACT_FET,  f_len)
        #self.linear_act_curr_img = nn.Linear(f_len + ACT_FET + LSTM_OUT_SIZE, f_len)

        self.linear_img_img = nn.Linear(f_len, SIM_SIZE)
        #self.lstm = nn.LSTM(LSTM_IN_SIZE, HIDDEN_SIZE, 1, batch_first=True)
        #self.linear_lstm_out = nn.Linear(HIDDEN_SIZE, LSTM_OUT_SIZE)

        #self.linear_fnext_out = nn.Linear(LSTM_OUT_SIZE, f_len)
        #self.linear_com_out = nn.Linear(LSTM_OUT_SIZE, COM_OUT)

        #self.linear_fnext_out = nn.Linear(HIDDEN_SIZE, f_len)
        #self.linear_com_out = nn.Linear(HIDDEN_SIZE, COM_OUT)
        #self.hidden = self.init_hidden()


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

        ''' 1. flatten mini-batch sequence of images'''
        f1 = self.cnn(I1.view(-1, 1, 106, 128))
        fg = self.cnn(Ig.view(-1, 1, 106, 128))
        ''' 2. flatten mini-batch sequence of action'''
        #fa0 = self.linear_act(a0.view(-1, 4))
        fa1 = self.linear_act(a1.view(-1, 4))

        #print f1.size() ## bs*seq x 8160
        #print fa0.size() ## bs*seq x 10

        ''' combine img and previous action feature to form one-step history'''
        #cat_f1_fa0 = torch.cat((f1, fa0), 1)
        cat_f1_fa1 = torch.cat((f1, fa1), 1)
        #print cat_f1_fa0.size()
        #lstm_inp = self.linear_act_prev_img(cat_f1_fa0)
        fnext = self.linear_act_img_curr(cat_f1_fa1)
        #print lstm_inp.size() ## 200 x 15

        ''' pack sequence to feed LSTM '''
        #lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)
        #lstm_inp = pack_padded_sequence(lstm_inp.view(bs, -1, LSTM_IN_SIZE),
        #        lengths, batch_first=True)
        ##print lstm_inp.data.size() ## 188 x 15
        #lstm_out, self.hidden = self.lstm(lstm_inp, self.hidden)
        #lstm_out, self.hidden = self.lstm(lstm_inp)

        ''' unpack sequence to feed in linear layer'''
        #print lstm_out.data.size() ## 188 x hidden_size
        #lstm_out_unpad, lengths_new = pad_packed_sequence(lstm_out, batch_first=True)
        #print lstm_out_unpad.size() ## bs x seq x hidden_size

        #lstm_out = self.linear_lstm_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        #com_out = self.linear_com_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        #fnext = self.linear_fnext_out(lstm_out_unpad.contiguous().view(-1, HIDDEN_SIZE))
        #com_out = self.linear_com_out(lstm_out)
        #print com_out.size() ## bs*seq x 2

        ''' combine img and current action to form one-step transition'''
        #combined = torch.cat([f1, fa1], 1)

        ''' combine transition with current estimate of internal state'''
        #combined = torch.cat([combined, lstm_out], 1)
        ''' get next internal state'''
        #fnext = self.linear_act_curr_img(combined)

        ''' evaluate similarity between target internal state & true internal state'''
        output = self.linear_img_img(torch.abs(fnext - fg))

        ''' squash value between (0, 1)'''

        sim = F.sigmoid(output)
        #com_out = F.sigmoid(com_out)

        #print sim.size() ## bs*seq x SIM_SIZE
        #print com_out.size() ## bs*seq x 2
        ''' pack into sequence for output'''
        sim_pack = pack_padded_sequence(sim.view(bs, -1, SIM_SIZE), lengths, batch_first=True)
        #com_pack = pack_padded_sequence(com_out.view(bs, -1, 2), lengths, batch_first=True)

        #print sim_pack.data.size() ## 188 x SIM_SIZE
        #print com_pack.data.size() ## 188 x 2

        #return sim_pack, com_pack
        #return sim
        return sim_pack

def get_num_parameters(model):
    model_parm = filter(lambda p:p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parm])

if __name__=='__main__':
    net = COM_net_sim(args.batch_size)
    print get_num_parameters(net)

    print net

#dot = make_dot(output)
#dot.render('network', view=True)



