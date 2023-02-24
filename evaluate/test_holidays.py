import os
import time

import h5py
import torch
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import transforms

#
#
# 这一部分代码是query_pyqt5抽取holidays数据集部分出来做检索实验的
# 遍历了可以查询的图像进行测试
#
#


if __name__ == '__main__':
    AP = 0
    recall_1 = 0
    queryNum = 0
    dataset_dir = "../dataset/Holidays"
    img_list = os.listdir(dataset_dir)
    #获取可以查询图像的list
    query_list = [img_name for img_name in img_list if img_name[5] == '0' and img_name[4] == '0']
    # for i in query_list:
    #     print(i)
    # print("l:" + str(len(query_list)))
    # model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = models.resnet50()
    model.load_state_dict(torch.load("../model/resnet50.pth"))   #加载本地模型
    model = nn.Sequential(*list(model.children())[:-1])   #去除全连接
    model.eval()
    #预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    start_time = time.time()
    #遍历查询图像
    for query in query_list:
        queryNum += 1
        group = query[1:4]
        group_start = img_list.index(query)
        group_num = 1
        #获取相似组图像张数
        for i in range(group_start + 1, len(img_list)):
            if img_list[i][1:4] == group:
                group_num += 1
            else:
                break
        for i in range(0, group_start):
            if img_list[group_start - i - 1][1:4] == group:
                group_num += 1
            else:
                break
        # print(group)
        # print(group_num)
        img_path = os.path.join(dataset_dir, query)
        #提取查询图像特征
        img = Image.open(img_path).convert("RGB")
        transformed_img = transform(img)
        processed_img = transformed_img.unsqueeze(0)
        input_img_feature = model(processed_img)
        local_db = "../model/holidays_features.h5"
        img_dict = {}
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)  #使用余弦相似度
        h5_file = h5py.File(local_db, 'r')
        #特征比对
        for img in img_list:
            local_db_feature = torch.from_numpy(h5_file[img][:])
            sim = cos_sim(input_img_feature, local_db_feature)
            img_dict[sim] = img
        h5_file.close()
        sim_list = sorted(img_dict.keys(), reverse=True)
        #计算指标
        num = 0
        right = 0
        print("=========query:" + str(queryNum))
        if img_dict[sim_list[0]][1:4] == group:
            recall_1 += 1
        else:
            print("img:" + img_path)
        for i in range(0, 3):
            res = img_dict[sim_list[i]][1:4]
            if res == group:
                right += 1
                num += right / (i + 1)
        AP += num / group_num
        mAP = AP / queryNum
        recall = recall_1 / queryNum
        print("mAP:" + str(mAP))
        print("r@1:" + str(recall))
    print("time spend:" + str(time.time() - start_time))