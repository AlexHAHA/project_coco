"""
    Author:xueyk
    Date:2019-11-06
"""
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

video_folder = r"D:\CETCA_DeepLearning\CETCA_UAVDataSet\zerotech_onboard_20190901\videos"
video_file = os.path.join(video_folder,r"MOV_00150 00_09_07-00_09_39.mp4")

dict_names_of_class = {0:'animal',1:'people',2:'car'}
dict_colors_of_class = {0:(0,255,255),1:(255,0,0),2:(0,255,0)}

'''
    图片预处理：将原图的右下角图片抠出来，输入到神经网络中
'''
def prepare_image1(img,in_dim):
    original_img = img
    # Convert opencv numpy to tensor
    # tensor dim:(batch, channel[RGB], height, weight)
    # opencv dim:(height,weight,channel[BGR])
    img_ = img[-in_dim:,-in_dim:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_,original_img

def add_layer1(img,detections,in_dim):
    h,w = img.shape[:2]
    color_area = (0,0,255)
    # detecting area
    offset_x = w-in_dim
    offset_y = h-in_dim
    cv2.rectangle(img, (offset_x, offset_y), (w,h), color_area, 2)
    detections = detections[0].numpy()
    #print(detections.shape)
    for i in range(detections.shape[0]):
        detect = detections[i]
        #print(detect.shape)
        p1 = (int(detect[0]) + offset_x, int(detect[1]) + offset_y)
        p2 = (int(detect[2]) + offset_x, int(detect[3]) + offset_y)

        cv2.rectangle(img, p1, p2,dict_colors_of_class[int(detect[-1])],2)
        #cv2.putText(img,label,())
    return img

'''
    图片预处理：将原图的整幅图片进行padding后缩放，作为神经网络的输入
'''
def prepare_image2(img,in_dim):
    original_img = img
    img_t   = img[:,:,::-1].transpose((2,0,1)).copy()
    img_t = torch.from_numpy(img_t).float().div(255.0)
    #print(img_t.dtype)
    img_t,_ = pad_to_square(img_t,0)
    img_t = resize(img_t, in_dim)
    img_t = img_t.unsqueeze(0)
    return img_t,original_img

'''
    直接在原始图片上叠加分类效果
'''
def add_layer2(img,detections,in_dim):
    h,w = img.shape[:2]
    #print(h,w)
    detections = detections[0].numpy()
    #print(detections.shape)
    diff_dim = int(np.abs(h - w)/2)
    scaler = (h / in_dim) if h>w else (w / in_dim)
    #print('scaler is: ',scaler)
    #print(detections)
    for i in range(detections.shape[0]):
        detect = detections[i]
        label = dict_names_of_class[int(detect[-1])]
        if h > w:
            p1 = (int(detect[0] * scaler) - diff_dim, int(detect[1] * scaler))
            p2 = (int(detect[2] * scaler) - diff_dim, int(detect[3] * scaler))
        else:
            p1 = (int(detect[0] * scaler), int(detect[1] * scaler) - diff_dim)
            p2 = (int(detect[2] * scaler), int(detect[3] * scaler) - diff_dim)
        cv2.rectangle(img, p1, p2, dict_colors_of_class[int(detect[-1])],2)
        cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                dict_colors_of_class[int(detect[-1])])
    return img

'''
    直接在网络输入图片上叠加分类效果
'''
def add_layer3(img,detections,in_dim):
    img = img.cpu().squeeze(0).numpy().transpose((1,2,0))
    img = img[:,:,::-1].copy()
    detections = detections[0].numpy()
    for i in range(detections.shape[0]):
        detect = detections[i]
        label = dict_names_of_class[int(detect[-1])]
        p1 = (int(detect[0]), int(detect[1]))
        p2 = (int(detect[2]), int(detect[3]))
        
        cv2.rectangle(img, p1, p2, dict_colors_of_class[int(detect[-1])],2)
        cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                dict_colors_of_class[int(detect[-1])])
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/cetca_samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    #parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_99.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/cetca_train/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")

    opt = parser.parse_args()
    print(opt)

    # choose GPU or CPU
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    #Tensor = torch.FloatTensor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    print("\nPerforming object detection:")
    prev_time = time.time()
    #
    frames = 0
    time_start = time.time()

    cap = cv2.VideoCapture(video_file)
    while cap.isOpened:
        ret, frame = cap.read()
        frame_height, frame_width = frame.shape[:2]
        frame = cv2.resize(frame,(int(frame_width/2),int(frame_height/2)),
                                interpolation=cv2.INTER_CUBIC)
        
        #
        img,origianl_img = prepare_image1(frame, opt.img_size)
        input_img = Variable(img.type(Tensor))
        #print(input_img.device)

        with torch.no_grad():
            # Yolov3-tiny: detections.size()=(1, 2535, 8), 2535=13*13*5
            detections = model(input_img)
            #print(f'-------{detections.size()}----------')
            #print(detections)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        if detections[0] is not None:
            origianl_img = add_layer1(origianl_img, detections, opt.img_size)
            process_img  = add_layer3(input_img, detections, opt.img_size)
            cv2.imshow("processimg",process_img)
            cv2.waitKey(1)
            #print(process_img.shape)
            #break
        cv2.imshow("video", origianl_img)
        key = cv2.waitKey(1)
        if key & 0xff == ord('q'):
            break

        # 计算处理速度
        frames += 1
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - time_start)))

