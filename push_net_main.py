'''
Main program to push objects

'''

__author__='Li Juekun'
__date__ = '2018/07/25'

## pytorch mods ##
import torch
import torch.nn as nn
import cv2
from push_net_model import *
from torch.autograd import Variable

import numpy as np
import os

import config as args
from img_utils import * ## utility function to manipulate images


''' Dimension of input image'''
W = 128.0 ##!!!! Important to make it float to prevent integer division becomes zeros
H = 106.0

''' Mode of Goal Configuration Specification'''
#MODE = 'xy' ## uncomment this line if you only care how to re-position an object
#MODE = 'w' ## uncomment this line if you only care how to re-orient an object
MODE = 'wxy' ## uncomment this line if care both re-position and re-orient an object

''' Method for comparison '''
METHOD = 'simcom' ## Original Push-Net
#METHOD = 'sim' ## Push-Net without estimating COM
#METHOD = 'nomem' ## Push-Net without LSTM

''' visualization options '''
CURR_VIS = True # display current image
NEXT_VIS = True # display target image
SAMPLE_VIS = False # display all sampled actions
BEST_VIS = True # display the best action


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

'''deep neural network predictor'''
class Predictor:
    def __init__(self):
        self.bs = args.batch_size
        model_path = 'model'
        best_model_name = args.arch[METHOD] + '.pth.tar'
        self.model_path = os.path.join(model_path, best_model_name)
        self.model = self.build_model()
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def build_model(self):
        if METHOD == 'simcom':
            return COM_net_sim(self.bs)
        elif METHOD == 'sim':
            return COM_net_sim_only(self.bs)
        elif METHOD == 'nomem':
            return COM_net_nomem(self.bs)

    def reset_model(self):
        ''' reset the hidden state of LSTM before pushing another new object '''
        self.model.hidden = self.model.init_hidden()

    def update(self, start, end, img_curr, img_goal):
        ''' update LSTM states after an action has been executed'''

        bs = self.bs
        A1 = []
        I1 = []
        Ig = []
        for i in range(bs):
            a1 = [[start[0]/W, start[1]/H, end[0]/W, end[1]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        if METHOD == 'simcom':
            sim_out, com_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)

    def evaluate_action(self, img_curr, img_goal, actions):
        ''' calculate the similarity score of actions '''
        bs = self.bs
        A1 = []
        I1 = []
        Ig = []

        for i in range(bs):
            a1 = [[actions[4*i]/W, actions[4*i+1]/H, actions[4*i+2]/W, actions[4*i+3]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        sim_out = None
        com_out = None

        if METHOD == 'simcom':
            sim_out, com_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)

        sim_np = sim_out.data.cpu().data.numpy()

        if MODE == 'wxy':
            sim_sum = np.sum(sim_np, 1) # measure (w ,x, y)
        elif MODE == 'xy':
            sim_sum = np.sum(sim_np[:,1:], 1) # measure (x, y)
        else:
            sim_sum = sim_np[:, 0] # measure (w)

        action_value = []
        for ii in range(len(sim_sum)):
            s = [actions[4 * ii], actions[4 * ii + 1]]
            e = [actions[4 * ii + 2], actions[4 * ii + 3]]
            action_value.append([[s, e], sim_sum[ii]])

        return action_value


''' Push Controller '''
class PushController:
    def __init__(self):
        self.num_action = args.num_action
        self.bs = args.batch_size
        ''' goal specification '''
        self.w = 30 # orientation in degree (positive in counter clockwise direction)
        self.x = 10 # x direction translation in pixel (horizontal axis of image plane)
        self.y = -10 # y direction translation in pixel (vertical axis of image plane)

        ## instantiate Push-Net predictor
        self.pred = Predictor()
        Ic = cv2.imread('test.jpg')[:,:,0]
        self.get_best_push(Ic.copy())


    def sample_action(self, img, num_actions):
        ''' sample [num_actions] numbers of push action candidates from current img'''
        s = 0.9
        safe_margin = 1.4
        out_margin = 2.0

        ## get indices of end push points inside object mask
        img_inner = cv2.resize(img.copy(), (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img_inner.shape
        img_end = np.zeros((int(H), int(W)))
        img_end[(int(H)-h)/2:(int(H)+h)/2, (int(W)-w)/2:(int(W)+w)/2] = img_inner.copy()
        (inside_y, inside_x) = np.where(img_end.copy()>0)

        ## get indices of start push points outside a safe margin of object
        img_outer1 = cv2.resize(img.copy(), (0,0), fx=safe_margin, fy=safe_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer1.shape
        img_start_safe = np.zeros((int(H), int(W)))
        img_start_safe = img_outer1.copy()[(h-int(H))/2:(h+int(H))/2, (w-int(W))/2:(w+int(W))/2]

        img_outer2 = cv2.resize(img.copy(), (0,0), fx=out_margin, fy=out_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer2.shape
        img_start_out = np.zeros((int(H), int(W)))
        img_start_out = img_outer2.copy()[(h-int(H))/2:(h+int(H))/2, (w-int(W))/2:(w+int(W))/2]

        img_start = img_start_out.copy() - img_start_safe.copy()
        (outside_y, outside_x) = np.where(img_start.copy()>100)

        num_inside = len(inside_x)
        num_outside = len(outside_x)

        actions = []
        for i in range(num_actions):
            start_x = 0
            start_y = 0
            end_x = 0
            end_y = 0
            while True:
                ## sample an inside point
                inside_idx = np.random.choice(num_inside)
                ## sample an outside point
                outside_idx = np.random.choice(num_outside)
                end_x = int(inside_x[inside_idx])
                end_y = int(inside_y[inside_idx])
                start_x = int(outside_x[outside_idx])
                start_y = int(outside_y[outside_idx])

                if start_x < 0 or start_x >= W or start_y < 0 or start_y >= H:
                    print 'out of bound'
                    continue
                if img[start_y, start_x] == 0:
                    break
                else:
                    continue

            actions.append(start_x)
            actions.append(start_y)
            actions.append(end_x)
            actions.append(end_y)

        return actions


    def get_best_push(self, Ic):
        ''' Input:
                Ic: current image mask
        '''
        img_in_curr = Ic.astype(np.uint8)

        _, img_in_curr = cv2.threshold(img_in_curr.copy(), 30, 255, cv2.THRESH_BINARY)

        ''' visualize current image '''
        if CURR_VIS:
            cv2.imshow('img', img_in_curr)
            cv2.waitKey(0)

        ''' generate goal image '''
        img_in_next = generate_goal_img(img_in_curr.copy(), self.w, self.x, self.y)

        ''' visualize goal image '''
        if NEXT_VIS:
            cv2.imshow('img', img_in_next)
            cv2.waitKey(0)

        ''' Sample actions '''
        actions = self.sample_action(img_in_curr.copy(), self.num_action)

        ''' visualize sampled actions '''
        if SAMPLE_VIS:
            for i in range(len(actions)/4):
                start = [actions[i*4], actions[i*4+1]]
                end = [actions[i*4+2], actions[i*4+3]]
                self.draw_action(img_in_curr.copy(), start, end, single=False)

        ''' Select actions '''
        num_action = len(actions) / 4
        num_action_batch = self.num_action / self.bs
        min_sim_score = 1000

        best_start = None
        best_end = None
        best_sim = None
        best_com = None

        action_batch = []
        hidden = None

        if not METHOD == 'nomem':
            hidden = self.pred.model.hidden

        action_value_pairs = []

        for i in range(num_action_batch):
            ## keep hidden state the same for all action batches during selection
            if not hidden == None:
                self.pred.model.hidden = hidden
            action = actions[4*i*self.bs: 4*(i+1)*self.bs]
            action_value = self.pred.evaluate_action(img_in_curr, img_in_next, action)
            action_value_pairs = action_value_pairs + action_value


        ''' sort action based on sim score '''
        action_value_pairs.sort(key=lambda x : x[1])

        ''' get best push action '''

        pack = action_value_pairs.pop(0)
        best_start = pack[0][0] ## best push starting pixel
        best_end = pack[0][1] ## best push ending pixel

        if BEST_VIS:
            self.draw_action(img_in_curr.copy(), best_start, best_end, single=True)

        ''' execute action '''
        ## TODO: do whatever to execute push action (best_start, best_end)

        ''' update LSTM hidden state '''
        self.pred.update(best_start, best_end, img_in_curr, img_in_next)


    def draw_action(self, img, start, end, single=True):

        (yy, xx) = np.where(img>0)
        img_3d = np.zeros((int(H), int(W), 3))
        img_3d[yy, xx] = np.array([255,255,255])

        sx = int(start[0])
        sy = int(start[1])
        ex = int(end[0])
        ey = int(end[1])

        cv2.line(img_3d, (sx, sy), (ex, ey), (0,0,255), 3)
        img_3d = img_3d.astype(np.uint8)

        cv2.imshow('action', img_3d)
        if single:
            ## draw the best action
            print 'press any key to continue ...'
            cv2.waitKey(0)
        else:
            ## draw all sample actions
            cv2.waitKey(10)


if __name__=='__main__':
    con = PushController()

