"""
mAP calculator
"""
import os
import sys
import glob
import shutil
import json
import operator

import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
    # 图片坐标系
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
    annotation: class, conf, bbox
    bbox: (left-top_x, left-top_y, right-bottom_x, right-bottom_y), the unit is pixel

"""
class MAP_Calculator(object):
    """
    需要设定目标文件夹的基本路径，其目录如下：
    folder(path_base):
    |-- input/
       |--ground_truth
       |--detection_results
       |--images
    |-- output/
       |--classes/
       |--temp/
       |--output.txt
    """
    MINOVERLAP = 0.5
    def __init__(self, path_mAP):
        self.path_base = path_mAP
        self.path_gt   = os.path.join(self.path_base, 'input', 'ground_truth')
        self.path_dr   = os.path.join(self.path_base, 'input', 'detection_results')
        self.path_img  = os.path.join(self.path_base, 'input', 'images')
        self.file_class_names = os.path.join(self.path_base, 'input', 'class.names')

        if not os.path.exists(self.path_gt):
            print("Error: no ground_truth folder exists")
        if not os.path.exists(self.path_dr):
            print("Error: no detection_results folder exists")
        if not os.path.exists(self.path_img):
            print("Error: no iamges folder exists")

        #创建output文件夹
        self.path_output = os.path.join(self.path_base, 'output')
        self.path_output_temp   = os.path.join(self.path_output, 'temp')
        self.path_output_class  = os.path.join(self.path_output, 'classes')
        if os.path.exists(self.path_output):
            shutil.rmtree(self.path_output)
        os.mkdir(self.path_output)
        os.mkdir(self.path_output_temp)
        os.mkdir(self.path_output_class)

        # 记录ground_truth每类的目标数量
        self.gt_counter_per_class = {}
        # 记录出现每类目标的图片数量，即一个类别共出现在多少张图片中
        self.gt_counter_images_per_class = {}
        # ground_truth中出现的所有目标的类别的idx
        self.gt_classes_idx = []

        # 记录detection_results每类目标的数量
        self.dr_counter_per_class = {}
        # detection_results中出现的所有目标的类别的idx
        self.dr_classes_idx = []
        
        self.class_names = self.load_classnames()
        #print(f"{len(self.class_names)}")
        #print(f"{self.class_names}")
        self.n_classes   = len(self.class_names)
        #print(self.class_names)
        self.ap_dictionary = {}

        # 记录每类检测结果为tp的目标数量
        self.count_true_positives = {}

        self.mAP = 0.0
        self.load_gt_data()
        self.load_dr_data()

    def load_classnames(self):
        with open(self.file_class_names,'r') as f:
            lines = f.readlines()
            class_names = [line.rstrip() for line in lines]
        return class_names

    def __load_txt(self, file_path):
        '''
        使用numpy加载txt，对只有一行的进行处理
        '''
        datas = np.loadtxt(file_path)
        if len(datas.shape) == 1:
            datas = np.expand_dims(datas, 0)
        return datas

    def load_gt_data(self):
        '''
        从ground_truth文件夹加载数据，得到：
        self.gt_counter_images_per_class
        '''
        #self.files_gt = os.listdir(self.path_gt)
        self.files_gt = glob.glob(self.path_gt + r'\*.txt')
        self.files_gt.sort()

        # 用于存放ground_truth信息，便于后续AP计算时使用
        self.gt_files_infos = {}
        for i, txt_file in enumerate(self.files_gt):
            file_id     = os.path.basename(txt_file).split(".txt",1)[0]
            txt_path_gt = os.path.join(self.path_gt, txt_file)
            txt_path_dr = os.path.join(self.path_dr, txt_file)
            # 判断与gt同名的dr文件是否存在
            if not os.path.exists(txt_path_dr):
                print(f"Error: corespond file {txt_file} not exist in detection_results folder")
                continue
            #
            already_seen_classes = []
            #
            lines = self.__load_txt(txt_path_gt)
            # 记录每个gt_file的信息
            infos = []
            for line in lines:
                class_idx = int(line[0])
                if class_idx in self.gt_counter_per_class:
                    self.gt_counter_per_class[class_idx] += 1
                else:
                    self.gt_counter_per_class[class_idx] = 1

                if class_idx not in already_seen_classes:
                    if class_idx in self.gt_counter_images_per_class:
                        self.gt_counter_images_per_class[class_idx] += 1
                    else:
                        self.gt_counter_images_per_class[class_idx] = 1
                    already_seen_classes.append(class_idx)
                infos.append({'idx':class_idx, 'bbox':list(line[1:]), 'used':False})
            #print(f"file={txt_file}")
            #print(self.gt_counter_images_per_class)

            self.gt_files_infos[file_id] = infos
            #print(f"file_id:{file_id}, infos:{infos}")
            #if i%10 == 0:
            #    break
            #print(f"{i},{txt_file}")
            #print(f"gt_counter_per_class:{self.gt_counter_per_class}")
            #print(f"gt_counter_images_per_class:{self.gt_counter_images_per_class}")

        #print(f'gt info={self.gt_files_infos}')
        self.gt_classes_idx = list(self.gt_counter_per_class.keys())
        self.n_classes      = len(self.gt_classes_idx)

    def load_dr_data(self):
        '''
        从detection_results文件夹中加载数据，获取每个类别所有目标的信息(confidence,bbox)，并按照confidence由大到小
        对目标信息进行排序，并将每个类别的目标信息存放到同名的.json文件中。
        '''
        self.files_dr = glob.glob(self.path_dr + r"\*.txt")
        self.files_dr.sort()

        # 先统计一下每个类别的目标数量
        for txt_file in self.files_dr:
            file_id = os.path.basename(txt_file).split(".txt",1)[0]
            txt_path_dr = os.path.join(self.path_dr, txt_file)
            lines = self.__load_txt(txt_path_dr)
            for line in lines:
                #print(f'file:{os.path.basename(txt_file)}')
                #print(f'line:{line}')
                tmp_class_name_idx = int(line[0])

                # 
                if tmp_class_name_idx in self.dr_counter_per_class:
                    self.dr_counter_per_class[tmp_class_name_idx] += 1
                else:
                    self.dr_counter_per_class[tmp_class_name_idx] = 1
        self.dr_classes_idx = list(self.dr_counter_per_class.keys())

        #把每个类别的所有目标的信息汇总起来
        for class_idx in self.gt_classes_idx:
            bounding_boxes = []
            for txt_file in self.files_dr:
                #print(f'txt_file={os.path.basename(txt_file)}')
                file_id = os.path.basename(txt_file).split(".txt",1)[0]
                txt_path_dr = os.path.join(self.path_dr, txt_file)
                txt_path_gt = os.path.join(self.path_gt, txt_file)

                if not os.path.exists(txt_path_gt):
                    print(f"Error: corespond file {txt_file} not exist in ground_truth folder")
                    continue
                #
                lines = self.__load_txt(txt_path_dr)
                for line in lines:
                    tmp_class_name_idx = int(line[0])
                    confidence         = float(line[1])
                    bbox               = list(line[2:])
                    #print(f'conf:={type(confidence)},bbox={type(bbox)}')

                    if tmp_class_name_idx == class_idx:
                        bounding_boxes.append({"confidence":confidence, "file_id":file_id, "bbox":bbox})
            #
            #print(f'{class_idx}:len is {len(bounding_boxes)}')
            # 根据bbox的confidence（置信度）对detection_results进行排序
            bounding_boxes.sort(key=lambda x:x['confidence'], reverse=True)
            with open(os.path.join(self.path_output_temp,f'{class_idx}_{self.class_names[class_idx]}_dr.json'), 'w') as f:
                json.dump(bounding_boxes, f)

    def ap_calculate(self):
        '''
         Calculate the AP for each class
         1. 图片中的每个gt_bbox只能被匹配一次，也就是说若有多个同类目标的dr_bbox都与一个gt_bbox匹配了，那么只有第一个匹配上的dr_bbox的tp=True，
            所以在调试的时候非常注意，每次本函数运行前都需要重新运行一下load_gt_data()以重新加载self.gt_files_infos变量。
        '''
        sum_AP = 0.0
        # open file to store the output
        with open(os.path.join(self.path_output, 'output.txt'), 'w') as output_file:
            output_file.write("# AP and precision/recall per class\n")
            # 
            for class_idx in self.gt_classes_idx:
                self.count_true_positives[class_idx] = 0
                class_name = self.class_names[class_idx]
                # Load detection-results of that class
                dr_file = os.path.join(self.path_output_temp, f'{class_idx}_{class_name}_dr.json')
                dr_data = json.load(open(dr_file))

                # Assign detection-results to ground-truth objects
                # nd代表class_idx类别的目标数量
                nd = len(dr_data)
                tp = [0] * nd
                fp = [0] * nd
                for idx, detection in enumerate(dr_data):
                    file_id = detection['file_id']
                    # open ground-truth with that file_id
                    gt_file = os.path.join(self.path_gt, file_id+'.txt')

                    #gt_datas = self.__load_txt(gt_file)
                    gt_datas = self.gt_files_infos[file_id]

                    ovmax = -1
                    gt_match = -1
                    # load detected object bounding-box
                    bbox_dr = detection['bbox']

                    #for gt_data in gt_datas()
                    for gt_data in gt_datas:
                        # look for a class index match
                        if gt_data['idx'] == class_idx:
                            #bbox_gt = gt_data[1:]
                            bbox_gt = gt_data['bbox']
                            bi = [max(bbox_dr[0],bbox_gt[0]), max(bbox_dr[1],bbox_gt[1]),
                                  min(bbox_dr[2],bbox_gt[2]), min(bbox_dr[3],bbox_gt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                ua = ((bbox_dr[2]-bbox_dr[0]+1) * (bbox_dr[3]-bbox_dr[1]+1) +
                                    (bbox_gt[2]-bbox_gt[0]+1) * (bbox_gt[3]-bbox_gt[1]+1) -
                                    iw * ih)
                                
                                ov = iw*ih / ua                         
                                if ov > ovmax:
                                    ovmax = ov 
                                    # 由于gt_data是dict类型，后续改变gt_match就会对gt_data进行改变
                                    gt_match = gt_data
    
                    # set minimum overlap
                    min_overlap = self.MINOVERLAP

                    # 计算TP,FP
                    if ovmax >= min_overlap:
                        if not bool(gt_match['used']):
                            gt_match['used'] = True
                            tp[idx] = 1
                            self.count_true_positives[class_idx] += 1
                            #print(f'match')
                        else:
                            pass
                            fp[idx] = 1
                            #print(f'repeat match')
                    else:
                        fp[idx] = 1
                        if ovmax > 0:
                            pass
                            #print('insufficient overlap')
                        else:
                            pass
                            #print(f'ovmax={ovmax}')

                # compute precision/recall
                cumsum = 0
                for idx, val in enumerate(fp):
                    fp[idx] += cumsum
                    cumsum += val
                cumsum = 0
                for idx, val in enumerate(tp):
                    tp[idx] += cumsum
                    cumsum += val
                #print(f'tp={tp}')
                #print(f'fp={fp}')
                rec = tp[:]
                for idx, val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / self.gt_counter_per_class[class_idx]
                prec = tp[:]
                for idx, val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

                ap, mrec, mprec = self.ap(rec[:], prec[:])
                sum_AP += ap

                #
                text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP"
                rounded_prec = ['%.2f' % elem for elem in prec]
                rounded_rec  = ['%.2f' % elem for elem in rec]
                output_file.write(text + "\n Precision: " + str(rounded_prec) + 
                                        "\n Recall: " + str(rounded_rec))
                output_file.write("\n")
                self.ap_dictionary[class_idx] = ap

                #
                # draw AP curve
                plt.plot(rec, prec, '-o')
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf() # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                #plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca() # gca - get current axes
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05]) # .05 to give some extra space
                # Alternative option -> wait for button to be pressed
                #while not plt.waitforbuttonpress(): pass # wait for key display
                # Alternative option -> normal display
                #plt.show()
                # save the plot
                fig.savefig(self.path_output + "\\classes\\" + class_name + ".png")
                plt.cla() # clear axes for next plot


            self.mAP = sum_AP / self.n_classes
            text = "mAP = {0:.2f}%".format(self.mAP*100)
            output_file.write(text + "\n")
        
    def draw_ground_truth_info(self):
        """
        Plot the total number of occurences of each class in the ground-truth
        保存为output/ground_truth_info.png
        """ 
        window_title = "ground_truth_info"
        plot_title   = "ground_truth\n"
        plot_title  += "(" + str(len(self.files_gt)) + "files and " + str(self.n_classes) + "classes)"
        x_label      = "Number of objects per class"
        output_path  = self.path_output + "/ground_truth_info.png"
        to_show      = False
        plot_color   = "forestgreen"

        #
        self.gt_counter_per_class_withnames = {self.class_names[key]:value for (key,value) in 
                                                self.gt_counter_per_class.items()}
        #
        self.draw_plot_func(
            self.gt_counter_per_class_withnames,
            self.n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
            )

    def draw_detection_results_info(self):
        """
        Plot the total number of occurences of each class in the "detection-results" folder
        and show the number of true positive
        保存为output/detection_results_info.png
        """
        window_title = "detection_results_info"
        # Plot title
        plot_title   = "detection_results\n"
        plot_title  += "(" + str(len(self.files_dr)) + " files and "
        cnt_nonzero  = sum(int(x) > 0 for x in list(self.dr_counter_per_class.values()))
        plot_title  += str(cnt_nonzero) + " detected classes)"
        # end Plot title
        x_label      = "Number of objects per class"
        output_path  = self.path_output + "/detection_results_info.png"
        to_show      = False
        plot_color   = 'forestgreen'
        #
        self.dr_counter_per_class_withnames = {self.class_names[key]:value for (key,value) in 
                                                self.dr_counter_per_class.items()}
        self.count_true_positives_withnames = {self.class_names[key]:value for (key,value) in 
                                                self.count_true_positives.items()}

        true_p_bar   = self.count_true_positives_withnames
        #
        self.draw_plot_func(
            self.dr_counter_per_class_withnames,
            len(self.dr_counter_per_class_withnames),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
            )

    def draw_map(self):
        """
        Draw mAP plot (Show AP's of all classes in decreasing order)
        保存为output/mAP.png
        """
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(self.mAP*100)
        x_label = "Average Precision"
        output_path = self.path_output + "/mAP.png"
        to_show = False
        plot_color = 'royalblue'
        #
        self.ap_dictionary_withnames = {self.class_names[key]:value for (key,value) in 
                                                self.ap_dictionary.items()}
        #
        self.draw_plot_func(
            self.ap_dictionary_withnames,
            self.n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    def ap(self, rec, prec):
        '''
        #Params
        rec: recall, list type
        prec: precision, list type
        '''
        rec.insert(0, 0.0)
        rec.append(1.0)
        mrec = rec[:]
        prec.insert(0, 0.0)
        prec.append(0.0)
        mpre = prec[:]

        # This part makes the precision monotonically decreasing（单调递减）
        # (goes from the end to the beginning)
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        # This part creates a list of indexes where the recall changes
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        # The Average Precision (AP) is the area under the curve
        # (numerical integration)
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre

    def adjust_axes(self, r, t, fig, axes):
        """
        Plot - adjust axes
        """
        # get text width for re-scaling
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        # get axis width in inches
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        # get axis limit
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1]*propotion])

    def draw_plot_func(self, dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
        """
        Draw plot using Matplotlib
        """    
        #print(dictionary)
        # sort the dictionary by decreasing value, into a list of tuples
        sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
        # unpacking the list of tuples into two lists
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)
        # 
        if true_p_bar != "":
            """
            Special case to draw in:
                - green -> TP: True Positives (object detected and matches ground-truth)
                - red -> FP: False Positives (object detected but does not match ground-truth)
                - pink -> FN: False Negatives (object not detected but present in the ground-truth)
            """
            fp_sorted = []
            tp_sorted = []
            for key in sorted_keys:
                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])
            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
            # add legend
            plt.legend(loc='lower right')
            """
            Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                # trick to paint multicolor with offset:
                # first paint everything and then repaint the first number
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values)-1): # largest bar
                    self.adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            """
            Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val) # add a space before
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                # re-set axes to show number inside the figure
                if i == (len(sorted_values)-1): # largest bar
                    self.adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title(window_title)
        # write classes in y axis
        tick_font_size = 12
        plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
        """
        Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height 
        top_margin = 0.15 # in percentage of the figure height
        bottom_margin = 0.05 # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title(plot_title, fontsize=14)
        # set axis titles
        # plt.xlabel('classes')
        plt.xlabel(x_label, fontsize='large')
        # adjust size of window
        fig.tight_layout()
        # save the plot
        fig.savefig(output_path)
        # show image
        if to_show:
            plt.show()
        # close the plot
        plt.close()        

#path_mAP = r"H:\deepLearning\dataset\visdrone\Task 1 - Object Detection in Images\VisDrone2019-DET-val\mAP"
path_mAP = r"H:\deepLearning\dataset\visdrone_mAP\map_truckerror"
if __name__ == '__main__':
    mc = MAP_Calculator(path_mAP)

    mc.ap_calculate()
    mc.draw_detection_results_info()
    mc.draw_ground_truth_info()
    mc.draw_map()

