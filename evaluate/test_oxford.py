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
# 这一部分代码是query_pyqt5抽取oxford数据集部分出来做检索实验的
# 遍历了可以查询的图像进行测试
#
#


if __name__ == '__main__':
    AP = 0
    recall_1 = 0
    queryNum = 0
    dataset_img_dir = "../dataset/Oxford/images"
    dataset_query_dir = "../dataset/Oxford/groundtruth"
    q_dir = "../dataset/Oxford/images_crop_196"
    img_list = os.listdir(dataset_img_dir)
    #q_list = os.listdir(q_dir)
    text_file_list = os.listdir(dataset_query_dir)
    #print(q_list)
    query_file_list = [text_file for text_file in text_file_list if text_file.endswith("query.txt")]
    #print(query_file_list)
    #获取可以查询的图像列表
    query_list=[]
    for query_file in query_file_list:
        with open(os.path.join(dataset_query_dir, query_file)) as f:
            query_list.append((f.readlines()[0].split(' ')[0][5:] +".jpg", query_file))
    # print(query_file_list)
    # print(len(query_list))
    #model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model = models.resnet50()
    model.load_state_dict(torch.load("../model/resnet50.pth"))  #加载本地模型
    model = nn.Sequential(*list(model.children())[:-1])  #去除全连接
    model.eval()
    #预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    start_time = time.time()
    #遍历可查询的图片
    for query, query_file in query_list:
        queryNum += 1
        group = query_file.replace("query", "good")
        with open(os.path.join(dataset_query_dir, group), 'r', encoding='utf-8') as f:
            group_num = len(f.readlines())
        img_path = os.path.join(q_dir, query)
        #提取查询图像特征
        img = Image.open(img_path).convert("RGB")
        transformed_img = transform(img)
        processed_img = transformed_img.unsqueeze(0)
        input_img_feature = model(processed_img)
        local_db = "../model/oxford_features.h5"
        img_dict = {}
        #特征比对
        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)  # 使用余弦相似度
        h5_file = h5py.File(local_db, 'r')
        for img in img_list:
            local_db_feature = torch.from_numpy(h5_file[img][:])
            sim = cos_sim(input_img_feature, local_db_feature)
            img_dict[sim] = img
        h5_file.close()
        sim_list = sorted(img_dict.keys(), reverse=True)
        #计算相关指标
        num = 0
        right = 0
        print("=========query:" + str(queryNum))
        print(query)
        with open(os.path.join(dataset_query_dir, group), 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line == img_dict[sim_list[0]].replace(".jpg", "")+ "\n":
                recall_1 += 1
                break
        for i in range(0, 3):
            group_search = img_dict[sim_list[i]].replace(".jpg", "")+ "\n"
            #print(group_search)
            for line in lines:
                # print(line)
                # print(group_search)
                if group_search == line :
                    right += 1
                    num += right / (i + 1)
                    break
        AP += num / group_num
        mAP = AP / queryNum
        recall = recall_1 / queryNum
        print("mAP:" + str(mAP))
        print("r@1:" + str(recall))
    print("time spend:" + str(time.time() - start_time))