# 在视频/网络摄像头上运行检测器
# 不在batch上迭代，而是在视频的帧上迭代

from __future__ import division
import pandas as pd
import operator
import shutil
import math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn import linear_model

# from playsound import playsound



# 命令行参数
def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    # images（用于指定输入图像或图像目录）
    parser.add_argument("--images", dest='images', help=
    "Image / Directory containing images to perform detection upon",
                        default="imgs", type=str)
    # det（保存检测结果的目录）
    parser.add_argument("--det", dest='det', help=
    "Image / Directory to store detections to",
                        default="det", type=str)
    # batch大小
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    # objectness置信度
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    # NMS阈值
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    # cfg（替代配置文件）
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    # reso（输入图像的分辨率，可用于在速度与准确度之间的权衡）
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)

    return parser.parse_args()


if __name__ == '__main__':
    V = {}
    V["blind"] = {}
    V["notBlind"] = {}
    V["notBlind"]["bool"] = 1
    V["blind"]["turn"] = 1
    #播放音频
    # pygame.init()
    # pygame.mixer.init()
    # car = pygame.mixer.Sound("sound/car.wav")
    # car.set_volume(0.2)
    # person = pygame.mixer.Sound("sound/person.wav")
    # person.set_volume(0.2)
    # Bicycle_wheel = pygame.mixer.Sound("sound/Bicycle_wheel.wav")
    # Bicycle_wheel.set_volume(0.2)
    # Fire_hydrant = pygame.mixer.Sound("sound/Fire_hydrant.wav")
    # Fire_hydrant.set_volume(0.2)
    # Motorcycle = pygame.mixer.Sound("sound/Motorcycle.wav")
    # Motorcycle.set_volume(0.2)
    # bus = pygame.mixer.Sound("sound/bus.wav")
    # bus.set_volume(0.2)
    # tree = pygame.mixer.Sound("sound/tree.wav")
    # tree.set_volume(0.2)
    # well_lid = pygame.mixer.Sound("sound/well_lid.wav")
    # well_lid.set_volume(0.2)
    # telegraph_pole = pygame.mixer.Sound("sound/telegraph_pole.wav")
    # telegraph_pole.set_volume(0.2)
    # left = pygame.mixer.Sound("sound/left.wav")
    # left.set_volume(0.2)
    # right = pygame.mixer.Sound("sound/right.wav")
    # right.set_volume(0.2)
    args = arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    num_classes = 10  # COCO数据集中目标的名称
    classes = load_classes("data/coco.names")

    # 初始化网络，加载权重
    print("正在加载网络QAQ")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("网络加载成功QvQ")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # GPU加速
    if CUDA:
        model.cuda()

    # 模型评估
    model.eval()

    # 绘制边界框:从colors中随机选颜色绘制矩形框
    # 边界框左上角创建一个填充后的矩形，写入该框位置检测到的目标的类别
    frontx1 = {}
    fronty1 = {}


    def write(x, results, i):


        c1 = tuple(x[1:3].int())  # 右上角坐标
        x1 = (x[1] + x[3]) / 2  # 边框x轴中间坐标
        y1 = (x[2] + x[3]) / 2
        frontx1[i] = int(x1)
        # fronty1[i] = int(x[4])
        fronty1[i] = int(y1)

        c2 = tuple(x[3:5].int())  # 左下角坐标
        img = results  # 仅处理一帧
        cls = int(x[-1])
        color = (255, 0, 0)

        # 随机设置颜色
        # color = random.choice(colors)
        label = "{0}".format(classes[cls])

        if label == 'blind tracks':
            df = pd.DataFrame({'x': frontx1, 'y': fronty1})
            X = pd.DataFrame(df['x'])
            Y = df['y']
            clf = linear_model.LinearRegression()
            clf.fit(X, Y)
            y_pred = clf.predict(X)
            data_x = np.array(X)
            data_y = np.array(y_pred)
            if i<100:
                V["blind"]["X"] = int(np.mean(X))
            else:
                V["blind"]["X"] = int(np.mean(X[i-100:i]))
            if i > 200:

                if int(np.max(X[i - 10:i])) - int(np.min(X[i - 10:i])) < 200:
                    cv2.arrowedLine(img, (int(np.mean(X[i - 100:i])), 700), (int(np.mean(X[i - 100:i])), 300), color,
                                    20)
                    V["blind"]["turn"] = 1
                else:
                    cv2.arrowedLine(img, (int(np.max(X[i - 10:i])), int(np.mean(y_pred[i - 200:i]))),
                                    (int(np.min(X[i - 10:i])), int(np.mean(y_pred[i - 200:i]))), color, 20)
                    if V["blind"]["turn"] ==1 and data_x[i-10]<data_x[i-1]:
                        print('左转弯')
                        V["blind"]["turn"] = 0
                        # playsound("sound/left.mp3")
                        # left.play()
                    elif data_x[i-10]>data_x[i-1] and V["blind"]["turn"] ==1 :
                        print('右转弯')
                        V["blind"]["turn"] = 0
                        # playsound("sound/right.mp3")
                        # right.play()
                    # cv2.arrowedLine(img, (200, int(np.mean(y_pred[i - 200:i]))),
                    #                 (400, int(np.mean(y_pred[i - 200:i]))), color, 20)

            elif i > 0 and i <= 200:
                if i <= 10:
                    if int(np.max(X)) - int(np.min(X)) < 200:
                        cv2.arrowedLine(img, (int(np.mean(X)), 700), (int(np.mean(X)), 300), color, 20)
                        V["blind"]["turn"] = 1
                    else:
                        cv2.arrowedLine(img, (int(np.max(X)), int(np.mean(y_pred))),
                                        (int(np.min(X)), int(np.mean(y_pred))), color, 20)
                        if V["blind"]["turn"]==1 and data_x[0]<data_x[i-1]:
                            print('左转弯')
                            V["blind"]["turn"] = 0
                            # playsound("sound/left.mp3")
                            # left.play()
                        elif data_x[0]>data_x[i-1] and V["blind"]["turn"] ==1:
                            print('右转弯')
                            V["blind"]["turn"] = 0
                            # playsound("sound/right.mp3")
                            # right.play()
                else:
                    if int(np.max(X[i - 10:i])) - int(np.min(X[i - 10:i])) < 200:
                        cv2.arrowedLine(img, (int(np.mean(X[i - 10:i])), 700), (int(np.mean(X[i - 10:i])), 300),
                                        color, 20)
                        V["blind"]["turn"] = 1
                    else:
                        cv2.arrowedLine(img, (int(np.max(X[i - 10:i])), int(np.mean(y_pred))),
                                        (int(np.min(X[i - 10:i])), int(np.mean(y_pred))), color, 20)
                        # if V["blind"]["turn"]==1:
                        # print(V["blind"]["turn"])
                        # if V["blind"]["turn"]==1 and int(X[i-10])<int(X[i]):
                        # print(len(X))
                        # print(X[i])
                        if V["blind"]["turn"]==1 and data_x[i-10]<data_x[i-1]:
                            print('左转弯')
                            V["blind"]["turn"] = 0
                            print(int(np.max(X[i - 10:i])))
                            print('--------------------')
                            print(int(np.min(X[i - 10:i])))
                            # playsound("sound/left.mp3")
                            # left.play()
                        elif data_x[i-10]>data_x[i-1] and V["blind"]["turn"] ==1:
                            print('右转弯')
                            V["blind"]["turn"] = 0
                            # playsound("sound/right.mp3")
                            # right.play()
                        # cv2.arrowedLine(img, (200, int(np.mean(y_pred))),
                        #                 (400, int(np.mean(y_pred))), color, 20)
                    # cv2.arrowedLine(img, (int(np.min(X[i-2:i])), int(np.max(y_pred[:i]))),(int(np.max(X[i-2:i])), int(np.min(y_pred[:i]))), color, 20)
                    # cv2.arrowedLine(img, (frontx1[0], int(np.max(y_pred))), (frontx1[i],int(np.min(y_pred)) ), color, 20)


        elif label != 'blind tracks':
            cv2.rectangle(img, c1, c2, color, 3)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]
            c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
            cv2.rectangle(img, c1, c2, color, -1)  # -1表示填充的矩形
            cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [225, 225, 225], 2)
            c3 = x[4].int()
            c4 = x[1].int()
            c5 = x[3].int()
            # print(V["notBlind"]["bool"])
            if c3 < 700 and c4<V["blind"]["X"] and c5>V["blind"]["X"] and V["notBlind"]["bool"]==1:
                if label=='Person':
                    print("前方有行人")
                    # person.play()
                    # playsound("sound/person.mp3")
                elif label == 'Bicycle wheel':
                    print("前方有自行车")
                    # Bicycle_wheel.play()
                    # playsound("sound/Bicycle_wheel.mp3")
                elif label == 'Fire hydrant':
                    print("前方有消防栓")
                    # playsound("sound/Fire_hydrant.mp3")
                    # Fire_hydrant.play()
                elif label == 'Motorcycle':
                    print("前方有摩托车")
                    # playsound("sound/Motorcycle.mp3")
                    # Motorcycle.play()
                elif label == 'Car':
                    print("前方有汽车")
                    # playsound("sound/car.mp3")
                    # car.play()
                elif label == 'Bus':
                    print("前方有公交车")
                    # playsound("sound/bus.mp3")
                    # bus.play()
                elif label == 'tree':
                    print("前方有树木")
                    # playsound("sound/tree.mp3")
                    # tree.play()
                elif label == 'well lid':
                    print("前方有井盖")
                    # playsound("sound/well_lid.mp3")
                    # well_lid.play()
                elif label == 'telegraph pole':
                    print("前方有电线杆")
                    # playsound("sound/telegraph_pole.mp3")
                    # telegraph_pole.play()
                V["notBlind"]["bool"]=0
            elif c3 > 700 or c4>V["blind"]["X"] or c5>V["blind"]["X"] and V["notBlind"]["bool"]==0 and i%17==0:
                V["notBlind"]["bool"]=1
        return img


    # 检测阶段
    videofile = "test.mp4"  # 加载视频文件路径

    save_path = videofile.split('.mp4')[0] + '/'
    is_exists = os.path.exists(save_path)
    if not is_exists:
        os.makedirs(save_path)
        # print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        # print('path of %s already exist and rebuild' % save_path)

    cap = cv2.VideoCapture(videofile)  # 用OpenCV打开视频/相机流

    fps = 23.3079
    # img_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = cap.read()
    videoWriter = cv2.VideoWriter(save_path + 'test' + '.avi', fourcc, fps, (frame.shape[1], frame.shape[0]))

    frames = 0  # 帧的数量
    start = time.time()
    # i=0
    # 在帧上迭代，一次处理一帧
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        # i=i+1
        if ret == True:
            img = prep_image(frame, inp_dim)
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            with torch.no_grad():
                # output = model(Variable(img, volatile=True), CUDA)
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("视频的FPS为 {:5.4f}".format(frames / (time.time() - start)))

                # 使用cv2.imshow展示画有边界框的帧
                # cv2.imshow("帧", frame)
                # print(frame)

                # save_name = save_path+'_'+str(i)+'.jpg'
                # cv2.imwrite(save_name,frame)
                # print('image of %s is saved' % save_name)
                videoWriter.write(frame)
                # print('image of %s is saved')

                show_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # plt.imshow(show_img)
                # plt.show()
                key = cv2.waitKey(1)
                # 用户按q，就会终止视频(代码中断循环)
                if key & 0xFF == ord('q'):
                    break
                continue

            output[:,1:5] = torch.clamp(output[:, 1:5], 0.0, float(inp_dim))
            im_dim = im_dim.repeat(output.size(0), 1) / inp_dim
            output[:, 1:5] *= im_dim

            classes = load_classes('data/coco.names')
            # colors = pkl.load(open("pallete", "rb"))

            list(map(lambda x: write(x, frame, count), output))
            count += 1
            # 使用cv2.imshow展示画有边界框的帧
            # cv2.imshow("帧", frame)
            # print(frame)
            # show_img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)

            # save_name = save_path + '_' + str(i) + '.jpg'
            # cv2.imwrite(save_name, frame)
            # print('image of %s is saved' % save_name)
            videoWriter.write(frame)
            # print('image of %s is saved')

            show_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # plt.imshow(show_img)
            # plt.show()
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(time.time() - start)
            print("视频的FPS为 {:5.4f}".format(frames / (time.time() - start)))
        else:
            break
    videoWriter.release()
    print('finish')




