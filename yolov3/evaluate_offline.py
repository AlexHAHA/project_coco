"""
对验证集进行检测，并将检测结果保存，后续使用mAP进行计算。
"""
from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import shutil
import datetime
import argparse

import pickle as pkl
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import cv2

def generate_colors():
    '''
    加载调色板
    '''
    current_path = os.getcwd()
    #print(current_path)
    pallete_path = os.path.abspath(os.path.join(current_path,".."))
    pallete_file = os.path.join(pallete_path,"resources\\pallete")
    colors = pkl.load(open(pallete_file, "rb"))
    return colors

class YoloDetect():
    def __init__(self, model_def,
                       data_config,
                       file_pretrained_weights,
                       path_detection_results,
                       img_size=416,
                ):
        self.flag_show     = False
        self.frame         = None
        self.detections    = [None]
        #
        data_config      = parse_data_config(data_config)
        self.valid_path  = data_config["valid"]
        self.class_names = load_classes(data_config["names"])
        self.path_detection_results = path_detection_results

        self.class_colors = generate_colors()
        self.img_size   = img_size
        self.conf_thres = 0.8
        self.nms_thres  = 0.4
        # choose GPU or CPU
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.model = Darknet(model_def).to(self.device)
            self.model.apply(weights_init_normal)
        except Exception as e:
            pass

        if file_pretrained_weights:
            # load user pretrained weights(checkpoint)
            if file_pretrained_weights.endswith(".pth"):
                self.model.load_state_dict(torch.load(file_pretrained_weights))
            # load others pretrained weights
            else:
                self.model.load_darknet_weights(file_pretrained_weights)
        self.model.eval()

    def prepare_image(self, img, in_dim):
        '''
        图片预处理：将原图的整幅图片进行padding后缩放，作为神经网络的输入
        '''
        original_img = img
        img_t   = img[:,:,::-1].transpose((2,0,1)).copy()
        img_t = torch.from_numpy(img_t).float().div(255.0)
        #print(img_t.dtype)
        img_t,_ = pad_to_square(img_t,0)
        img_t = resize(img_t, in_dim)
        img_t = img_t.unsqueeze(0)
        return img_t,original_img

    def add_layer1(self, img, detections, in_dim):
        '''
        直接在原始图片上叠加目标识别和分类效果
        img:       原始图片
        detection: yolov3输出的bounding box
        in_dim:    yolov3输出图片大小
        '''
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
            label = self.class_names[int(detect[-1])]
            if h > w:
                p1 = (int(detect[0] * scaler) - diff_dim, int(detect[1] * scaler))
                p2 = (int(detect[2] * scaler) - diff_dim, int(detect[3] * scaler))
            else:
                p1 = (int(detect[0] * scaler), int(detect[1] * scaler) - diff_dim)
                p2 = (int(detect[2] * scaler), int(detect[3] * scaler) - diff_dim)
            cv2.rectangle(img, p1, p2, self.class_colors[int(detect[-1])],4)
            cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                                    self.class_colors[int(detect[-1])], 2)
        return img

    def add_layer2(self, img, detections, in_dim):
        '''
        直接在网络输入图片上叠加目标识别和分类效果
        '''
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

    def detect(self,frame):
        '''
        基础识别功能函数，输入frame，输出识别结果
        '''
        #print(f"detect.py,detect: {frame.shape}")
        img, original_img = self.prepare_image(frame, self.img_size)
        #cv2.imshow('prepare img', img)
        #cv2.waitKey(200)
        input_img = Variable(img.type(self.Tensor))

        detections = None
        with torch.no_grad():
            detections = self.model(input_img)
            #print(f"detect.py,detect: detections.size {detections.size()}")
            #print(f"detect.py,detect: {detections[0]}")
            detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
        return detections

    def get_img_bboxs(self, img, detections, in_dim):
        '''
        获取在原始图片上对应的bbox，并且叠加检测效果
        Args:
            img:       原始图片
            detection: yolov3输出的bounding box
            in_dim:    yolov3输出图片大小
        Returns:
            img: 叠加bbox后的图片
            bbox: 检测结果
        '''
        h,w = img.shape[:2]
        #print(h,w)
        detections = detections[0].numpy()
        #print(detections.shape)
        diff_dim = int(np.abs(h - w)/2)
        scaler = (h / in_dim) if h>w else (w / in_dim)
        #print('scaler is: ',scaler)
        #print(detections)
        obj_numbers = detections.shape[0]
        bboxs = np.zeros((obj_numbers, 6))
        for i in range(obj_numbers):
            detect = detections[i]
            label = self.class_names[int(detect[-1])]
            if h > w:
                p1 = (int(detect[0] * scaler) - diff_dim, int(detect[1] * scaler))
                p2 = (int(detect[2] * scaler) - diff_dim, int(detect[3] * scaler))
            else:
                p1 = (int(detect[0] * scaler), int(detect[1] * scaler) - diff_dim)
                p2 = (int(detect[2] * scaler), int(detect[3] * scaler) - diff_dim)
            cv2.rectangle(img, p1, p2, self.class_colors[int(detect[-1])],2)
            cv2.putText(img, label, p1, cv2.FONT_HERSHEY_PLAIN, 1.0,
                                    self.class_colors[int(detect[-1])], 2)
            bboxs[i] = np.array([int(detect[-1]), detect[-2], p1[0], p1[1], p2[0], p2[1]])
        return img, bboxs

    def save_txt(self, f_name, datas, fmts):
        with open(f_name, 'w') as f:
            for x in datas:
                line = "{} {:.2f} {} {} {} {}\n".format(int(x[0]),x[1], int(x[2]), int(x[3]), int(x[4]), int(x[5]))
                f.write(line)
            '''
            for x in datas:
                line = ""
                for i,fmt in enumerate(fmts):
                    line += fmt.format(x[i])
                line += "\n"
                f.write(line)
            '''

    def evalute1(self):
        """
        对文件夹中的一张张图片进行检测（推理inference），
        并保存结果至detect_results文件夹中
        """
        img_files = []
        with open(self.valid_path) as f:
            img_files = f.readlines()
            img_files = [x.rstrip() for x in img_files]
        #print(img_files)
        for img_file in img_files:
            #print(img_file)
            img_name = os.path.basename(img_file)
            # 保存inference后的bbox
            dr_bbox_file  = os.path.join(self.path_detection_results, 'labels', 
                                    img_name.replace('.png','.txt').replace('.jpg','.txt'))
            # 保存inference后叠加bbox的图像
            dr_img_file   = os.path.join(self.path_detection_results, 'images', img_name)
            img = cv2.imread(img_file)
            detections = self.detect(img)
            if detections[0] is not None:
                img, bboxs = self.get_img_bboxs(img, detections, self.img_size)
                cv2.imwrite(dr_img_file, img)
                #np.savetxt(dr_bbox_file, bboxs, delimiter=' ', fmt='%d')
                self.save_txt(dr_bbox_file, bboxs, ["{:d}","{:.2f}","{:d}","{:d}","{:d}","{:d}"])
                if self.flag_show:
                    cv2.imshow('detecting result', img)
                    cv2.waitKey(100)
            else:
                #print(f"evaluate1-detections: {detections}")
                pass

    def evalute2(self, pic_path):
        '''
        使用dataloader加载图片，每次只加载一张进行识别。
        '''
        
        self.model.eval()
        dataloader = DataLoader(
                    ImageFolder(pic_path, img_size=self.img_size),
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
        )
        
        #prev_time      = time.time()
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            print(f'batch_i:{batch_i}')
            #print(f"type:{type(input_imgs)},shape:{input_imgs.shape}")
            print(f"img_path:{img_paths}")

            #
            img_name = os.path.basename(img_paths[0])
            # 保存inference后的bbox
            dr_bbox_file = os.path.join(self.path_detection_results, 'labels',
                                        img_name.replace('.jpg','.txt').replace('.png','.txt'))
            # 保存inference后叠加bbox的图像
            dr_img_file   = os.path.join(self.path_detection_results, 'images', img_name)
            # debug
            frame = cv2.imread(img_paths[0])
            #cv2.imshow('cv read', frame)
            #input_img = input_imgs[0].numpy().transpose((1,2,0)).copy()
            #input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
            #cv2.imshow('dataload', input_img)
            #cv2.waitKey(10000)
            #return
            
            # Configure input
            input_imgs = Variable(input_imgs.type(self.Tensor))
            # Get detections
            detections = []
            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
                #print(detections)
            # Log progress
            #current_time = time.time()
            #fps          = 1/(current_time-prev_time)
            #prev_time    = current_time
            # Save image and detections
            if detections[0] is not None:
                img, bboxs = self.get_img_bboxs(frame, detections, self.img_size)
                #img        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(dr_img_file, img)
                #np.savetxt(dr_bbox_file, bboxs, delimiter=' ', fmt='%d')
                self.save_txt(dr_bbox_file, bboxs, ["{:d}","{:.2f}","{:d}","{:d}","{:d}","{:d}"])
                if self.flag_show:
                    cv2.imshow('res', img)
                    cv2.waitKey(10000)
            return

path_weights     = r"H:\deepLearning\dataset\COCO2014\yolov3\weights\yolov3.weights"
path_class_names = r"H:\deepLearning\dataset\COCO2014\yolov3\classes.names"
path_model_def   = r"config\yolov3.cfg"
path_data_config = r"config\coco2014_val.data"
path_detection_results = r"H:\deepLearning\dataset\COCO2014\yolov3\map_yolov3\detection_results"

def test1():
    if os.path.exists(path_detection_results):
        shutil.rmtree(path_detection_results)
    os.mkdir(path_detection_results)
    os.mkdir(os.path.join(path_detection_results, 'labels'))
    os.mkdir(os.path.join(path_detection_results, 'images'))
    
    yolodetect = YoloDetect(model_def=path_model_def,
                            data_config=path_data_config,
                            path_detection_results=path_detection_results,
                            file_pretrained_weights=path_weights)
    yolodetect.evalute1()

pic_path = r"H:\deepLearning\dataset\COCO2014\val2014\val2014"
def test2():
    if os.path.exists(path_detection_results):
        shutil.rmtree(path_detection_results)
    os.mkdir(path_detection_results)
    os.mkdir(os.path.join(path_detection_results, 'labels'))
    os.mkdir(os.path.join(path_detection_results, 'images'))
    
    yolodetect = YoloDetect(model_def=path_model_def,
                            data_config=path_data_config,
                            path_detection_results=path_detection_results,
                            file_pretrained_weights=path_weights)
    yolodetect.evalute2(pic_path)

if __name__ == "__main__":
    test1()
    

