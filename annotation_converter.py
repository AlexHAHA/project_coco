"""
Author: alex
Func:
    convert annotation between comman defines
"""
import os
import numpy as np
import cv2

xyxy_2_xywh = 0
xywh_2_ccwh = 1
ccwh_2_xyxy = 2
xyxy_2_ccwh = 3
xywh_2_xyxy = 4
ccwh_2_xywh = 5

unit_p2p   = 0
unit_p2n   = 1  # pixel to normlized
unit_n2p   = 2  # normlized to pixel
unit_n2n   = 3

class AnnotationConver(object):
    """
    xyxy to xywh
    xyxy to 

    unit_normlized2pixel
    unit_pixel2normlized
    """
    def __init__(self, path_source_anno, path_source_img, path_target_anno, 
                 bbox_ct, unit_ct, columns):
        '''
        bbox_ct: bbox convert type
        unit_ct: unit convert type
        '''
        self.path_source_anno = path_source_anno
        self.path_source_img  = path_source_img
        self.path_target_anno = path_target_anno
        self.columns = columns
        self.bbox_ct = bbox_ct
        self.unit_ct = unit_ct
        self.bbox_fun = [self.xyxy_2_xywh, self.xywh_2_ccwh, self.ccwh_2_xyxy,
                         self.xyxy_2_ccwh, self.xywh_2_xyxy, self.ccwh_2_xywh]

    def convert(self):
        anno_files = os.listdir(self.path_source_anno)
        for anno_file in anno_files:
            bbox = np.loadtxt(os.path.join(self.path_source_anno, anno_file))
            # 如果annotation文件中只有一行或什么都没有则应该进行相应处理。
            if len(bbox.shape) == 1:
                bbox = np.expand_dims(bbox, 0)
            elif len(bbox.shape) == 0:
                print(f"blank file:{os.path.basename(anno_file)}")
            img_file = os.path.join(self.path_source_img, os.path.basename(anno_file).replace('.txt','.jpg'))
            img = cv2.imread(img_file)
            img_size = img.shape[:2]
            #print(f"imgsize={img_size}")
            #print(f"anno_file={os.path.basename(anno_file)}")
            #print(f"befor:bbox={bbox}")
            if self.unit_ct == unit_n2p:
                #print(f"img_size={img_size}")
                bbox = self.unit_normlized2pixel(bbox, img_size)
            bbox = self.bbox_fun[self.bbox_ct](bbox)

            fmt = '%d'
            if self.unit_ct == unit_p2n:
                fmt = '%.3f'
                bbox = self.unit_pixel2normlized(bbox, img_size)
            #print(f"after:bbox={bbox}")

            dr_file = os.path.join(self.path_target_anno, anno_file)
            np.savetxt(dr_file, bbox, fmt=fmt)
            

    def xyxy_2_xywh(self, bbox):
        width  = bbox[:, self.columns[2]] - bbox[:, self.columns[0]] + 1
        height = bbox[:, self.columns[3]] - bbox[:, self.columns[1]] + 1
        bbox[:, self.columns[2]] = width
        bbox[:, self.columns[3]] = height
        return bbox

    def xywh_2_ccwh(self, bbox):
        cx = bbox[:, self.columns[0]] + bbox[:, self.columns[2]] // 2 + 1
        cy = bbox[:, self.columns[1]] + bbox[:, self.columns[3]] // 2 + 1
        bbox[:, self.columns[0]] = cx
        bbox[:, self.columns[1]] = cy
        return bbox

    def ccwh_2_xyxy(self, bbox):
        x1 = bbox[:, self.columns[0]] - bbox[:, self.columns[2]] // 2
        y1 = bbox[:, self.columns[1]] - bbox[:, self.columns[3]] // 2
        x2 = bbox[:, self.columns[0]] + bbox[:, self.columns[2]] // 2
        y2 = bbox[:, self.columns[1]] + bbox[:, self.columns[3]] // 2 
        bbox[:, self.columns[0]] = x1
        bbox[:, self.columns[1]] = y1
        bbox[:, self.columns[2]] = x2
        bbox[:, self.columns[3]] = y2
        return bbox  

    def xyxy_2_ccwh(self, bbox):
        bbox = self.xyxy_2_xywh(bbox)
        bbox = self.xywh_2_ccwh(bbox)
        return bbox

    def xywh_2_xyxy(self, bbox):
        bbox = self.xywh_2_ccwh(bbox)
        bbox = self.ccwh_2_xyxy(bbox)
        return bbox

    def ccwh_2_xywh(self, bbox):
        bbox = self.ccwh_2_xyxy(bbox)
        bbox = self.xyxy_2_xywh(bbox)
        return bbox

    def unit_normlized2pixel(self, bbox, img_size):
        img_width  = img_size[1]
        img_height = img_size[0]
        bbox[:, self.columns[0]] *= img_width
        bbox[:, self.columns[2]] *= img_width
        bbox[:, self.columns[1]] *= img_height
        bbox[:, self.columns[3]] *= img_height
        bbox = bbox.astype(np.int32)
        return bbox

    def unit_pixel2normlized(self, bbox, img_size):
        img_width  = img_size[1]
        img_height = img_size[0]
        bbox[:, self.columns[0]] /= img_width
        bbox[:, self.columns[2]] /= img_width
        bbox[:, self.columns[1]] /= img_height
        bbox[:, self.columns[3]] /= img_height
        bbox = bbox.astype(np.float32)
        return bbox

def test1():
    path_source_anno = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-val\labels"
    path_source_img  = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-val\images"
    path_target_anno = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-val\target_xyxy"
    bbox_ct = ccwh_2_xyxy
    unit_ct = unit_n2p
    columns = [1,2,3,4]
    ac = AnnotationConver(path_source_anno,path_source_img,path_target_anno,
                          bbox_ct, unit_ct, columns)
    ac.convert()   

def test2():
    path_source_anno = r"H:\deepLearning\dataset\visdrone_mAP\map_truckerror\input\detection_results_yolo"
    path_source_img  = r"H:\deepLearning\dataset\visdrone_mAP\map_truckerror\input\images"
    path_target_anno = r"H:\deepLearning\dataset\visdrone_mAP\map_truckerror\input\detection_results"
    bbox_ct = ccwh_2_xyxy
    unit_ct = unit_n2p
    columns = [1,2,3,4]
    ac = AnnotationConver(path_source_anno,path_source_img,path_target_anno,
                          bbox_ct, unit_ct, columns)
    ac.convert()        

if __name__ == '__main__':
    test2()
