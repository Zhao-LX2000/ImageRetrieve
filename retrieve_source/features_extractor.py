import os

import h5py
import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torchvision.transforms import transforms
from PIL import Image

def extract_model():
    #net = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)   #获取模型
    net = models.resnet50()    #获取完模型之后找到模型路径，复制到项目路径下，便于在其他电脑环境下运行
    net.load_state_dict(torch.load("../model/resnet50.pth"))
    #print(net)
    net = nn.Sequential(*list(net.children())[:-1])   #保留网络到倒数第一层（不包括），也就是去除全连接层
    # net.fc = nn.Linear(2048, 2048)
    # nn.init.eye(net.fc.weight)
    # for param in net.parameters():
    #     param.requires_grad = False
    return net


def extractor(imgs_path, output_path, model):
    print("=======================feature extraction starts===========================")
    #预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    model.cuda().eval()   #使用GPU
    features = {}   #(图片名字，对应特征)
    #遍历抽取特征
    for file_name in os.listdir(imgs_path):
        if file_name.endswith(".jpg"):
            img_path = os.path.join(imgs_path, file_name)
            img = Image.open(img_path).convert("RGB")
            transformed_img = transform(img)
            processed_img = transformed_img.unsqueeze(0).cuda()  #加一维度
            feature = model(processed_img).cpu()
            features[file_name] = feature.data.numpy()   #加进数据
    print("=======================features writing starts===========================")
    #将特征字典写入HDF5文件中保存在本地
    h5_file = h5py.File(output_path, 'w')
    for file_name in features:
        h5_file.create_dataset(file_name, data=features[file_name])
    h5_file.close()
    print("=======================features writing over===========================")

if __name__ == '__main__':
    model = extract_model()
    extractor("../dataset/Holidays", "../model/holidays_features.h5", model)
    extractor("../dataset/Oxford/images", "../model/oxford_features.h5", model)