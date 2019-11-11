import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from configparser import ConfigParser  #  配置类，专门来读取配置文件
import os

#读入训练集和验证集；标签可以在train.py中直接拿出计算loss，且和图片一一匹配，所以没有重新创建文件保存；

cf = ConfigParser()

#自动寻找地址：
path = os.getcwd()
path = os.path.dirname(path)
# print(path)
path = path+'\conf\data_input.conf'

cf.read(path)
secs = cf.sections()

# #检验
# print(secs)
#
# options = cf.options("Address")  # 获取某个section名为Address所对应的键，仅名字
# print(options)
#
# items = cf.items("Address")  # 获取section名为Address所对应的全部键值对，名字和值都有
# print(items)
#
# host = cf.get("Value", "batch_size")  # 获取[Address]中cifar10对应的值，特定名字所对应的值
# print(host)


##########################################################################################################
def dataset_input(dataset_type):
    dir_train = cf.get("Address", dataset_type)
    dir_val = cf.get("Address", dataset_type)

    #这里可以删除
    transformation = transforms.Compose([
            #将图片的像素点范围由[0, 255]转化为[0.0, 1.0],并且变为tensor
            transforms.ToTensor(),
            #([0, 1] - 0.5) / 0.5 = [-1, 1]，将像素点范围更改为[-1, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


    if  dataset_type == "cifar10":
            train_dataset = datasets.CIFAR10(dir_train, train=True, transform=transformation,
                                   download=True)
            val_dataset = datasets.CIFAR10(dir_val, train=False, transform=transformation,
                                   download=True)
    else:
            if dataset_type == "cifar100":
                train_dataset = datasets.CIFAR100(dir_train, train=True, transform=transformation,
                                                 download=True)
                val_dataset = datasets.CIFAR100(dir_val, train=False, transform=transformation,
                                                download=True)
            else:
                print("dataset loading input has some issues.")

    batch_size = cf.get("Value", "batch_size")
    train_db = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_db = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_db, val_db


