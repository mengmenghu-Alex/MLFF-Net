import numpy as np
import argparse
from Net.model import *
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join
from Utils.uqim_utils import getUIQM
from Utils.ssm_psnr_utils import getSSIM, getPSNR
import csv

def get_ssim_psnr(input_image_path, label_iamge_path):
    """
        - label_iamge_path contain ground-truths
        - input_image_path contain generated images
    """
    label_iamge_paths = sorted(glob(join(label_iamge_path, "*.*")))
    input_image_paths = sorted(glob(join(input_image_path, "*.*")))
    ssims, psnrs = [], []
    for label_path, input_path in zip(label_iamge_paths, input_image_paths):
        label_name = basename(label_path).split('.')[0] #获取图像名字
        input_name = basename(input_path).split('.')[0]
        if (label_name == input_name):
            g_im = Image.open(input_path)
            r_im = Image.open(label_path)
            r_im = r_im.resize((g_im.size[0],g_im.size[1]))
            ssim = getSSIM(np.array(r_im), np.array(g_im))
            ssims.append(ssim)
            r_im = r_im.convert("L")
            g_im = g_im.convert("L")
            psnr = getPSNR(np.array(r_im), np.array(g_im))
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)

def get_uiqm(image_path):
    paths = sorted(glob(join(image_path, "*.*")))
    uqims = []
    for img_path in paths:
        im = Image.open(img_path)
        uiqm = getUIQM(np.array(im))
        uqims.append(uiqm)
    return np.array(uqims)

def get_mse(input_image_path, label_iamge_path):
    input_paths = sorted(glob(join(input_image_path, "*.*")))
    label_paths = sorted(glob(join(label_iamge_path, "*.*")))
    mses = []
    for label_path, input_path in zip(label_paths, input_paths):
        gtr_f = basename(label_path).split('.')[0]
        gen_f = basename(input_path).split('.')[0]
        if (gtr_f == gen_f):
            input = Image.open(label_path)
            label = Image.open(input_path)
            input = input.resize((label.size[0],label.size[1]))
            assert input.size == label.size
            input_np = np.array(input)
            label_np = np.array(label)  
            squared_error = np.sum((input_np.astype("float") - label_np.astype("float")) ** 2)
            mse = squared_error/(float(input_np.shape[1]*input_np.shape[0])*3)
            mses.append(mse)

    return np.array(mses)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_images_path', type=str, default="./Evaluation/GT/",help="Ground truth")
    parser.add_argument('--enhance_images_path', type=str, default='./Results/',help="Enhanced Results Path")
    parser.add_argument('--method', type=str, default="CLUIE-Net")
    parser.add_argument('--dataset', type=str, default='LSUI')

    config = parser.parse_args()
    input_image_path = join(join(config.enhance_images_path,config.method),config.dataset)
    label_image_path = join(config.label_images_path,config.dataset)

    SSIM, PSNR = get_ssim_psnr(input_image_path,label_image_path)
    MSE = get_mse(input_image_path,label_image_path)
    UIQM = get_uiqm(input_image_path)
       
    print("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM), np.std(SSIM)))
    print("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR), np.std(PSNR)))
    print("MSE >> Mean: {0} std: {1}".format(np.mean(MSE), np.std(MSE)))
    print("UIQM >> Mean: {0} std: {1}".format(np.mean(UIQM), np.std(UIQM)))