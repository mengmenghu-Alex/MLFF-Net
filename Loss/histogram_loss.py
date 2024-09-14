import cv2
import numpy as np
import torch.nn as nn

def calc_histogram(image):
    total_histr = 0
    total_histg = 0
    total_histb = 0
    # 将图像从PyTorch张量转换为NumPy数组
    for i in range (image.shape[0]):
        # L1_loss = nn.L1Loss()
        # inp = np.squeeze((input[i,:,:,:]).cpu().detach().numpy(),axis=0)
        # lab = np.squeeze((label[i,:,:,:]).cpu().detach().numpy(),axis=0)
        inp = (image[i,:,:,:]).cpu().detach().numpy()
        inp = np.transpose(inp,(1,2,0))
        hist_r = cv2.calcHist([inp], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([inp], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([inp], [2], None, [256], [0, 256])
        total_histr = total_histr + hist_r
        total_histg = total_histg + hist_g
        total_histb = total_histb + hist_b
    return total_histr, total_histg, total_histb

def histogram_matching_loss(original, processed):
    original_hist_r, original_hist_g, original_hist_b = calc_histogram(original)
    processed_hist_r, processed_hist_g, processed_hist_b = calc_histogram(processed)
    
    # 使用相关性比较直方图
    loss_r = cv2.compareHist(original_hist_r, processed_hist_r, cv2.HISTCMP_BHATTACHARYYA)
    loss_g = cv2.compareHist(original_hist_g, processed_hist_g, cv2.HISTCMP_BHATTACHARYYA)
    loss_b = cv2.compareHist(original_hist_b, processed_hist_b, cv2.HISTCMP_BHATTACHARYYA)
    
    # 将相关性转换为损失
    return loss_r + loss_g + loss_b


class his(nn.Module):
    def __init__(self):
        super(his,self).__init__()

    def forward(self,input,label):
        return histogram_matching_loss(input,label)


