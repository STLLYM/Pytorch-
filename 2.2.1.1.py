#coding=gbk
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch
import numpy as np
from sklearn.datasets import load_boston#����2
from PIL import Image
from glob import glob
boston = load_boston()

x = torch.rand(10)
print(x.size())#��������0ά��������������ֻ��һ��Ԫ�ص�һά������ʾ
print("**************************1")
temp = torch.FloatTensor([23,24,24.5,26,27.2,23.0])
print(temp.size())#��������1ά����������һ��Ԫ�����е�����
print("***************************2")
boston_tensor = torch.from_numpy(boston.data)
print(boston.data.shape)
print(boston_tensor.size())#(2ά����)��������ṹ�����ݿɱ�ʾΪ������
print(boston_tensor[:2])#����Ϊboston�������ϣ�λ�ڰ�sticik_learn��
print(boston_tensor[:10:5])
print("****************************3")
panda = np.array(Image.open('D:\\pytorch�鼮Դ��\\DeepLearningwithPyTorch_Code\\DLwithPyTorch-master\\images\\panda.jpg').resize((224, 224)))
panda_tensor = torch.from_numpy(panda)
print(panda_tensor.size())
plt.imshow(panda)
print("*****************************4")
sales1 = torch.FloatTensor([1000.0, 323.2, 333.4, 444.5, 1000.0, 323.2, 333.4, 444.5])
sales2 = torch.FloatTensor([1000.0, 323.2, 333.4, 444.5, 1000.0, 323.2, 333.4, 444.5])
print(sales1[:5])#��Ƭ�����������г���n��֮ǰ��Ԫ�أ���ʽ��������[:n]
print(sales2[:-5])
plt.imshow(panda_tensor[:, :, 0].numpy())#�������ֱ�Ϊֻȡͼ��һ��ͨ�����ϣ��Ͳü���èͷ�����£�
plt.imshow(panda_tensor[25:175,60:130,0].numpy())
print("******************************5")
#�˴�Ϊ���д��룬����·���ɸ��Ե��Ծ���
#4ά�������ͳ�������Ϊ��ͼ��������Ϊ
data_path='/Users/vishnu/Documents/fastAIPytorch/fastai/courses/dl1/data/dogscats/train/cats/'
cats = glob(data_path+'*.jpg')
#��ͼƬת��Ϊnumpy����
cat_imgs = np.array([np.array(Image.open(cat).resize((224,224))) for cat in cats[:64]])
cat_imgs = cat_imgs.reshape(-1,224,224,3)
cat_tensors = torch.from_numpy(cat_imgs)
print(cat_tensors.size())
print("*******************************6")
