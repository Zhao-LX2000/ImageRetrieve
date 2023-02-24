import os
import sys
import time
import h5py
import torch
from PIL import Image
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QGroupBox, QHBoxLayout, QComboBox, QLabel
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import transforms

def val_model():
    #net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)   #获取模型
    net = models.resnet50()    #获取完模型之后找到模型路径，复制到项目路径下，便于在其他电脑环境下运行
    net.load_state_dict(torch.load("../model/resnet50.pth"))
    #print(net)
    #print('===========================')
    net = nn.Sequential(*list(net.children())[:-1])   #保留网络到倒数第一层（不包括），也就是去除全连接层
    # net.fc = nn.Linear(2048, 2048)
    # nn.init.eye(net.fc.weight)
    # for param in net.parameters():
    #     param.requires_grad = False
    #print(net)
    return net

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1300, 700)          #最外层窗口
        self.setWindowTitle("图像检索")
        self.init_ui()
        self.input_img = -1     #输入图像，初始化为-1便于判断用户是否输入图像
        #self.lastButton = -1
        self.AP = 0    #初始化累加AP值
        self.recall_1 = 0   #初始化累加r@1
        self.queryNum = 0    #初始化查询次数
        # 以下是获取oxford数据集的（查询图片，相似组文件(*good.txt)）对应关系
        self.query_group_dic = {}
        self.query_file_dir = "../dataset/Oxford/groundtruth"
        text_file_list = os.listdir(self.query_file_dir)
        #过滤出groundtruth文件夹中query.text
        query_file_list = [text_file for text_file in text_file_list if text_file.endswith("query.txt")]
        for query_file in query_file_list:
            #读取第一行，获取第一行第一个，处理成和数据集对应图像一样的名字
            with open(os.path.join(self.query_file_dir, query_file)) as f:
                self.query_group_dic[f.readlines()[0].split(' ')[0][5:] + ".jpg"] = query_file.replace("query", "good")
        #print(self.query_group_dic)
    def init_ui(self):
        container = QVBoxLayout() #最外层盒子 垂直布局
        #---------------------------------------------------------
        retrieve_res_box = QGroupBox("检索结果") #上层组件
        self.lbl = QLabel() #图片1
        self.lbl.setFixedSize(300, 300)
        self.lbl2 = QLabel() #图片2
        self.lbl2.setFixedSize(300, 300)
        self.lbl3 = QLabel() #图片3
        self.lbl3.setFixedSize(300, 300)
        retrieve_res_layout = QHBoxLayout()  #上层组件布局 水平布局
        retrieve_res_layout.addWidget(self.lbl)
        retrieve_res_layout.addWidget(self.lbl2)
        retrieve_res_layout.addWidget(self.lbl3)
        retrieve_res_box.setLayout(retrieve_res_layout)
        #---------------------------------------------------------
        down_widget = QWidget() #下层窗口
        three_hori = QHBoxLayout() #下层组件布局 水平布局
        three_hori_1 = QGroupBox("输入图像") #三个组件盒子
        three_hori_2 = QGroupBox("信息")
        three_hori_3 = QGroupBox("响应")
        #-------------------------左
        three_hori.addWidget(three_hori_1)
        v_layoutx = QHBoxLayout()
        self.lblx = QLabel()   #查询图片
        self.lblx.setFixedSize(300, 300)
        v_layoutx.addWidget(self.lblx)
        three_hori_1.setLayout(v_layoutx)
        # -------------------------中
        three_hori.addWidget(three_hori_2)
        v_layout = QVBoxLayout()
        v_layout2 = QHBoxLayout()
        nei = QGroupBox()
        self.label6 = QLabel(self)
        self.label6.setText('提示:请选择图片')
        label = QLabel(self)
        label.setText('请选择需要检索的数据集：')
        self.combo = QComboBox(self)
        self.combo.addItem("Holidays")
        self.combo.addItem("Oxford")
        self.combo.currentIndexChanged.connect(self.reset)   #重置事件
        btn = QPushButton("打开图片")
        btn.setFixedHeight(100)
        btn4 = QPushButton("开始检索")
        btn4.setFixedHeight(100)
        btn4.clicked.connect(self.retrieve)    #索引事件
        btn.clicked.connect(self.openImg)   #按键事件
        label2 = QLabel(self)
        label2.setText('by Zhao Lingxiang')
        label2.setAlignment(QtCore.Qt.AlignRight)  #右对齐
        nei.setLayout(v_layout2)
        v_layout.addWidget(self.label6)
        v_layout2.addWidget(label)
        v_layout2.addWidget(self.combo)
        v_layout.addWidget(nei)
        v_layout.addWidget(btn)
        v_layout.addWidget(btn4)
        v_layout.addWidget(label2)
        three_hori_2.setLayout(v_layout)
        # -------------------------右
        three_hori.addWidget(three_hori_3)
        v_layout3 = QVBoxLayout()
        self.label3 = QLabel(self)
        self.label3.setText('time spend: ')
        self.label4 = QLabel(self)
        self.label4.setText('mAP:')
        self.label5 = QLabel(self)
        self.label5.setText('r@1:')
        v_layout3.addWidget(self.label3)
        v_layout3.addWidget(self.label4)
        v_layout3.addWidget(self.label5)
        three_hori_3.setLayout(v_layout3)
        down_widget.setLayout(three_hori)
        #---------------------------------------------------
        container.addWidget(retrieve_res_box)
        container.addWidget(down_widget)
        self.setLayout(container)

    # 更换数据集绑定的重置事件，需要清零一些参数
    def reset(self):
        self.AP = 0
        self.recall_1 = 0
        self.queryNum = 0
        self.label3.setText('time spend:')
        self.label4.setText('map:')
        self.label5.setText('r@1:')
        self.label6.setText("提示:请重新选择图片。")
        self.input_img = -1


    # 获取图像对应组和组内相似图像张数
    def getInfo(self, filename, checked_index):
        # 以下是Holidays数据集处理部分
        if checked_index == 0:
            base = filename.split('/')[-1]
            group = base[1:4]    #组号为Holidays数据集图片名字的第二、三、四个字符，对于101000.jpg，组号就是010
            path = filename.replace(base, "")
            path_list = os.listdir(path)
            group_start = path_list.index(base)
            group_num = 1
            #从查询图片在数据集的字典序位置开始，向前向后遍历，查看有多少张组号相同的图片
            for i in range(group_start + 1, len(path_list)):
                if path_list[i][1:4] == group :
                    group_num += 1
                else:
                    break
            for i in range(0, group_start):
                if path_list[group_start - i - 1][1:4] == group :
                    group_num += 1
                else:
                    break
        #以下是Oxford数据集处理部分
        elif checked_index == 1:
            print("====================")
            #print(filename.split('/')[-1])
            #利用输入图片名字和之前构建的字典，可以找到对应的good.txt作为group
            group = os.path.join(self.query_file_dir, self.query_group_dic[filename.split('/')[-1]])
            #只需要知道*good.txt文件里有多少行就知道有多少相似图片
            with open(os.path.join(group), 'r', encoding='utf-8') as f:
                group_num = len(f.readlines())
            # with open(filename, 'r', encoding='utf-8') as f:  # 打开文件
            #     lines = f.readlines()
            #     if len(lines) != 1:
            #         return
            #     img_file = os.path.join("../dataset/Oxford/images", lines[0].split(' ')[0][5:] +".jpg")
            print(group_num)
        return group, group_num

    def openImg(self, arg):
        checked_index = self.combo.currentIndex()
        #弹出文件上传窗口
        self.input_img, filetype = QtWidgets.QFileDialog.getOpenFileName(None, "lll", os.getcwd(),"query jpg Files(*.jpg);;All Files(*)")
        if not self.input_img.endswith(".jpg"):   #判断是否为jpg，不是则返回
            return
        self.label6.setText("提示:可以进行检索了")
        self.group, self.group_num = self.getInfo(self.input_img, checked_index)
        ##print(self.input_img + ":" + self.group)
        pixmap = QPixmap(self.input_img)  # 按指定路径找到图片
        pixmapx = pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio)
        self.lblx.setPixmap(pixmapx)  # 在label上显示图片
        #self.lblx.setScaledContents(True)  # 让图片自适应label大小
        #预处理块
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        #获取模型
        model = val_model()
        model.eval()
        #预处理
        img = Image.open(self.input_img).convert("RGB")
        transformed_img = transform(img)
        processed_img = transformed_img.unsqueeze(0)  #加一维度
        #获取特征
        self.input_img_feature = model(processed_img)


    #检索
    def retrieve(self, arg):
        if self.input_img == -1 :
            self.label6.setText("提示:您还没有输入图片，请选择图片。")
            return
        start = time.time()   #开始计时
        checked_index = self.combo.currentIndex()
        #根据选择数据集设置数据集地址和本地特征数据库地址
        if checked_index == 0:
            img_path = "../dataset/Holidays/"
            local_db = "../model/holidays_features.h5"
        elif checked_index == 1:
            img_path = "../dataset/Oxford/images/"
            local_db = "../model/oxford_features.h5"
        else:
            print('System Wrong.')
            return
        img_list = os.listdir(img_path)
        img_dict = {}
        # 使用余弦相似度，对第二维度进行计算
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
        # 遍历本地特征数据库，计算余弦相似度
        h5_file = h5py.File(local_db, 'r')
        for img in img_list:
            local_db_feature = torch.from_numpy(h5_file[img][:])
            sim = cos_sim(self.input_img_feature, local_db_feature)
            img_dict[sim] = img
        h5_file.close()
        sim_list = sorted(img_dict.keys(), reverse=True)  #根据余弦相似度进行排序
        # print(img_dict[sim_list[0]])
        # print(img_dict[sim_list[1]])
        # print(img_dict[sim_list[2]])
        pixmap = QPixmap(os.path.join(img_path, img_dict[sim_list[0]]))  # 按指定路径找到图片
        pixmap2 = QPixmap(os.path.join(img_path, img_dict[sim_list[1]]))  # 按指定路径找到图片
        pixmap3 = QPixmap(os.path.join(img_path, img_dict[sim_list[2]]))  # 按指定路径找到图片
        self.lbl.setPixmap(pixmap.scaled(300, 300, QtCore.Qt.KeepAspectRatio))  # 在label上显示图片
        self.lbl2.setPixmap(pixmap2.scaled(300, 300, QtCore.Qt.KeepAspectRatio))  # 在label上显示图片
        self.lbl3.setPixmap(pixmap3.scaled(300, 300, QtCore.Qt.KeepAspectRatio))  # 在label上显示图片
        self.label3.setText('time spend: %s s' % (time.time() - start))
        self.queryNum += 1   #加一次全局查询次数
        #Holidays的计算指标
        if checked_index == 0:
            num = 0
            right = 0
            #如果第一张结果图像在相似组里r@1累加1
            if img_dict[sim_list[0]][1:4] == self.group:
                self.recall_1 += 1
            for i in range(0, 3) :
                res = img_dict[sim_list[i]][1:4]
                #print(res)
                #print(self.group)
                #计算AP值计算的分子
                if res == self.group :
                    right += 1
                    num += right / (i + 1)
        # Oxford的计算指标
        elif checked_index == 1:
            num = 0
            right = 0

            with open(self.group, 'r') as f:
                lines = f.readlines()
            for i in range(0, 3):
                group_search = img_dict[sim_list[i]].replace(".jpg", "")+ "\n"
                #print(group_search)
                for line in lines:
                    if group_search == line :
                        right += 1
                        num += right / (i + 1)
                        if i == 0 :
                            self.recall_1 += 1          # 如果第一张结果图像在相似组里r@1累加1
                        break
        self.AP += num / self.group_num    #除以相似组图片张数，获得AP
        mAP =  self.AP / self.queryNum     #AP除以查询次数，获得mAP
        recall = self.recall_1 / self.queryNum  #累计r@1除以查询次数，获得平均的r@1
        self.label4.setText('map: %f' % mAP)
        self.label5.setText('r@1: %f' % recall)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWindow()
    w.show()
    app.exec()  #循环执行