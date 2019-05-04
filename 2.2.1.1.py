#coding=gbk
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np
from sklearn.datasets import load_boston#用于2
from PIL import Image
from glob import glob
boston = load_boston()

x = torch.rand(10)
print(x.size())#标量，（0维张量），可以用只含一个元素的一维张量表示
print("**************************1")
temp = torch.FloatTensor([23,24,24.5,26,27.2,23.0])
print(temp.size())#向量，（1维张量），是一个元素序列的数组
print("***************************2")
boston_tensor = torch.from_numpy(boston.data)
print(boston.data.shape)
print(boston_tensor.size())#(2维向量)，大多数结构化数据可表示为表或矩阵
print(boston_tensor[:2])#资料为boston房价资料，位于包sticik_learn中
print(boston_tensor[:10:5])
print("****************************3")
panda = np.array(Image.open('D:\\pytorch书籍源码\\DeepLearningwithPyTorch_Code\\DLwithPyTorch-master\\images\\panda.jpg').resize((224, 224)))
panda_tensor = torch.from_numpy(panda)
print(panda_tensor.size())
plt.imshow(panda)
print("*****************************4")
sales1 = torch.FloatTensor([1000.0, 323.2, 333.4, 444.5, 1000.0, 323.2, 333.4, 444.5])
sales2 = torch.FloatTensor([1000.0, 323.2, 333.4, 444.5, 1000.0, 323.2, 333.4, 444.5])
print(sales1[:5])#切片张量，用于切出第n个之前的元素，形式：张量名[:n]
print(sales2[:-5])
plt.imshow(panda_tensor[:, :, 0].numpy())#这两个分别为只取图像一个通道（上）和裁剪熊猫头部（下）
plt.imshow(panda_tensor[25:175,60:130,0].numpy())
print("******************************5")
#此处为书中代码，具体路径由各自电脑决定
#4维张量类型常见例子为批图像处理，作用为
data_path='/Users/vishnu/Documents/fastAIPytorch/fastai/courses/dl1/data/dogscats/train/cats/'
cats = glob(data_path+'*.jpg')
#将图片转换为numpy数组
cat_imgs = np.array([np.array(Image.open(cat).resize((224,224))) for cat in cats[:64]])
cat_imgs = cat_imgs.reshape(-1,224,224,3)
cat_tensors = torch.from_numpy(cat_imgs)
print(cat_tensors.size())
print("*******************************6")
