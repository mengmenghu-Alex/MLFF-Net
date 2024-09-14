import torch.nn as nn
import cv2
import numpy as np

def color_loss(input,label):
    total_loss = 0
    
    for i in range (input.shape[0]):
        # L1_loss = nn.L1Loss()
        # inp = np.squeeze((input[i,:,:,:]).cpu().detach().numpy(),axis=0)
        # lab = np.squeeze((label[i,:,:,:]).cpu().detach().numpy(),axis=0)
        inp = (input[i,:,:,:]).cpu().detach().numpy()
        inp = np.transpose(inp,(1,2,0))
        labe = (label[i,:,:,:]).cpu().detach().numpy()
        labe = np.transpose(labe,(1,2,0))
        input_lab = cv2.cvtColor(inp,cv2.COLOR_BGR2LAB)
        label_lab = cv2.cvtColor(labe,cv2.COLOR_BGR2LAB)

        input_yuv = cv2.cvtColor(inp,cv2.COLOR_BGR2YUV)
        label_yuv = cv2.cvtColor(labe,cv2.COLOR_BGR2YUV)

        inp_l, inp_a, inp_b = cv2.split(input_lab)
        lab_l, lab_a, lab_b = cv2.split(label_lab)

        inp_y,inp_cr,inp_cb = cv2.split(input_yuv)
        lab_y,lab_cr,lab_cb = cv2.split(label_yuv)
       
        a_loss = np.mean(np.abs(inp_a - lab_a))
        b_loss = np.mean(np.abs(inp_b - lab_b))
        cr_loss = np.mean(np.abs(inp_cr - lab_cr))  #l1loss定义实现
        cb_loss = np.mean(np.abs(inp_cb - lab_cb))
        total_loss = total_loss + a_loss + b_loss + cr_loss + cb_loss
    
    # return total_loss/input.shape[0]
    return total_loss/(128.0*input.shape[0])

class Color(nn.Module):
    def __init__(self):
        super(Color,self).__init__()

    def forward(self,input,label):
        return color_loss(input,label)