[P1. PyTorch环境的配置及安装（Configuration and Installation of PyTorch)【PyTorch教程】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1hE411t7RN?p=1&vd_source=339a4744bd362ae7b381fd9629bfd3a9)



# Dataset加载数据
```python
import os
from torch.utils.data import Dataset
import torchvision
from PIL import Image


demo3：实现自己的dataset类
继承的类，必须实现父类的所有接口/重写方法
继承dataset就是要实现/重写__init__  __getitem__
'''
# class MyData(Dataset):
#
#     def __init__(self):
#         pass
#
#     def __getitem__(self, item):
#         pass
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        # 设root dir=dataset/train
        self.root_dir = root_dir
        # 设label dir=ants
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        # if label dir = ants, then path=dataset/train/ants
        # os.listdir could list all ants img 的名字 如0013035.jpg
        # img path是一个list
        self.img_path = os.listdir(self.path)
    def __getitem__(self, idx):
        """
        对MyData对象使用索引操作就会自动来到这个函数下边，双下划线是python中的魔法函数
        :param idx:
        :return:
        """
        img_name = self.img_path[idx]
        # 名字是0013035.jpg的图片的路径
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label
    # 再写一个获取数据集长度的魔法函数
    def __len__(self):
        return len(self.img_path)
'''
demo4:获取蚂蚁数据集dataset
'''
root_dir = "./dataset/train"
label_dir = "ants"
ants_dataset = MyData(root_dir, label_dir)
# print(ants_dataset[0])
# # output:(<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x28C66665EE0>, 'ants')
# img, label = ants_dataset[0]
# img.show()
# print(label)
'''
demo5：再来获取蜜蜂的数据集
'''
root_dir = "./dataset/train"
label_dir = "bees"
bees_dataset = MyData(root_dir, label_dir)
'''
demo6:dataset数据集拼接
'''
train_dataset = ants_dataset + bees_dataset
```

# TensorBoard的使用
```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tu_demo1 import MyData
'''
demo1:summary writer的介绍
Writes entries directly to event files in the log_dir to be consumed by TensorBoard. 
SummaryWriter把实体直接写入一个目录里边，这个目录可以被TensorBoard读取，然后画成图
SummaryWriter常用只有1个参数log_dir
Example：
    传入log dir参数
    writer = SummaryWriter("my_experiment")
    # folder location: my_experiment，比如后边常写的logs
    comment参数可以给log dir后边加后缀
    writer = SummaryWriter(comment="LR_0.1_BATCH_16")
    # folder location: runs/May04_22-14-54_s-MacBook-Pro.localLR_0.1_BATCH_16/
    #                  runs/May04_22-14-54_s-MacBook-Pro.local是默认的目录
'''
# writer = SummaryWriter("logs")
# writer.add_image()
# writer.add_scalar()
# writer.close()
# tensorboard --logdir=logs 这是启动命令
'''
demo2：写标量数据
tag (string): Data identifier 标识符，表名
scalar_value (float or string/blobname): Value to save  y轴
global_step (int): Global step value to record          x轴
'''
# writer = SummaryWriter("logs")
#
# for i in range(100):
#     writer.add_scalar("y=x summary", i, i)
#     writer.add_scalar("y=2x summary", 2*i, i)
#
# writer.close()
'''
demo3：写图片数据
tag (string): Data identifier
img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
global_step (int): Global step value to record
写图片，要求图片必须是上面的三种类型，我们使用numpy来改变图片数据类型
使用numpy array会报错，因为你的图片维度顺序不符合writer的要求，报错如下：
TypeError: Cannot handle this data type: (1, 1, 512)
改正如下，使用dataformats='HWC'向writer说明我们的三位顺序是HWC：
writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')
可以在控制台简单运行一下代码，就知道顺序了
'''
# writer = SummaryWriter("logs")
#
# image_path = "dataset/train/ants/0013035.jpg"
# img_pil = Image.open(image_path)
# img_array = np.array(img_pil)
#
# # writer.add_image("img test", img_array, 1)
# writer.add_image("img test", img_array, 1, dataformats='HWC')
# writer.close()
'''
demo4:练习使用图片写入
'''
# 获取数据集
root_dir = "./dataset/train"
label_dir = "ants"
ants_dataset = MyData(root_dir, label_dir)
# 写入board
step = 1
writer = SummaryWriter("logs")
for img, label in ants_dataset:
    img_array = np.array(img)
    print(type(img_array), img_array.shape)
    print("正在写入第{}张图片".format(step))
    writer.add_image("ants images", img_array, step, dataformats='HWC')
    step = step + 1
# 最后一定关闭writer
writer.close()
```


# Transforms
```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
'''
demo1:tosensor简单使用 
tosensor：
Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor
example:
    pil --> tensor
    tensor([[[0.3137, 0.3137, 0.3137,  ..., 0.3176, 0.3098, 0.2980],
             [0.3176, 0.3176, 0.3176,  ...,
'''
# 获取pil类型图片
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
# 创建需要的transforms工具，并给工具起名字
tensor_trans = transforms.ToTensor()
# 使用工具
tensor_img = tensor_trans(img)
print(tensor_img)
"""
demo2:为什么需要tensor数据类型
因为tensor包含了一些属性是计算神经网络是必不可少的
grad：梯度
device：设备
is CUDA：
requires grad：保留梯度
"""
# tensor_img.grad = 0
# tensor_img.requires_grad = False
'''
demo3:使用tensor数据类型写入board
'''
writer = SummaryWriter("logs")
writer.add_image('tensor img', tensor_img, 1)
writer.close()
```

```python
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
'''
demo1:魔法函数
实例化一个对象之后，如果这个对象可以不用.方法名，而是可以直接传参数，那说明这个对象就有一个对应的魔法函数
比如getitem call len
'''
img_path = "dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
writer = SummaryWriter('logs')
'''
demo2:compose的用法
compose就是一个指定一个transform操作序列，定义了一条加工流水线
Example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.PILToTensor(),
    >>>     transforms.ConvertImageDtype(torch.float),
    >>> ])
'''
'''
demo3:normalize的用法
计算方法：output[channel] = (input[channel] - mean[channel]) / std[channel]
说人话：该像素上的值减去均值，再除以方差
'''
trans_norm = transforms.Normalize([5, 0.5, 0.5], [0.5, 0.5, 0.5])
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
img_norm = trans_norm(img_tensor)
writer.add_image('norm img', img_norm)
# 在board中可以看到norm之后图片变化很大，因为故意设的均值很大
'''
demo4:resize的用法
Resize the input image to the given size.
注意如果给了一个int就是变为正方形，给（H，W）才是H W
resize不会改变图片的数据类型
'''
trans_resize = transforms.Resize((512, 500))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
print(img_resize)
'''
demo5:random crop
随机剪一个指定尺寸的新图片
'''
trans_randomcrop = transforms.RandomCrop(500)
trans_compose = transforms.Compose([
    trans_randomcrop,
    trans_tensor
])
for i in range(10):
    img_crop_tensor = trans_compose(img)
    writer.add_image('crop tensor', img_crop_tensor, i)
writer.close()
```

# torchvision中的数据集使用
```python
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
'''
demo1:使用torchvision获取数据集
前面是数据的类型，后面是图片的标签 数字对应类别
(<PIL.Image.Image image mode=RGB size=32x32 at 0x22F00A48F10>, 6)
类别：
['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
'''
# train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
# test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)
# print(train_set[0])
# print(train_set.classes)
# img, target = train_set[0]
# print(train_set.classes[target])
'''
demo2:使用compose对数据集做transform操作
'''
dataset_trans = transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_trans, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_trans, download=True)
writer = SummaryWriter('logs')
for i in range(10):
    img, target = train_set[i]
    writer.add_image('test torchvison compose', img, i)
writer.close()
```

# DataLoader的使用
```python
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
'''
demo1:dataloader的简单使用
batch size：loader能每次装弹4枚进入枪膛，或者理解每次抓四张牌
shuffle：每次epoch是否打乱原来的顺序，就像打完一轮牌后，洗不洗牌
drop last：最后的打他不够一个batch 还要不要了
'''
train_set = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_set = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=True, num_workers=0, drop_last=False)
img, target = test_set[0]
print(img.shape)
print(target)
# 使用board可视化
writer = SummaryWriter("logs")
step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    # 这里如果你用add image会报错，因为这个方法只能一次写一个图片，你必须换add images方法来写入带有批处理的图片
    # writer.add_image("test set loader", imgs, step)
    writer.add_images("test set loader", imgs, step)
    step = step + 1
writer.close()
```

# nn.Module
```python
import torch
from torch import nn
"""
demo1:写一个最基本的神经网络都要哪些骨头
两个骨头就是骨架：__init__ forward
"""
class Tudui(nn.module):
    def __init__(self):
        super(Tudui, self).__init__()
    def forward(self, input):
        output = input + 1
        return output
tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)
print(output)
```

# 卷积
```python
import torch
import torch.nn.functional as F
"""
demo1:使用conv2d
input kernel:都是四维
stride：步长
padding：如果步长是1，又想保持输入输出高宽不变，就把padding设置1
"""
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
# conv2d需要输入的tensor是四维的（batch， c，h，w），但是现在的input kernel是二维
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))
output = F.conv2d(input, kernel, stride=1)
print(output)
# tensor([[[[10, 12, 12],
#           [18, 16, 16],
#           [13,  9,  3]]]])
output2 = F.conv2d(input, kernel, stride=1, padding=1)
print(output2)
# tensor([[[[ 1,  3,  4, 10,  8],
#           [ 5, 10, 12, 12,  6],
#           [ 7, 18, 16, 16,  8],
#           [11, 13,  9,  3,  4],
#           [14, 13,  9,  7,  4]]]])
```

# 卷积 池化 非线性激活 线性层
```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, ReLU, Sigmoid, Linear, Flatten, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
"""
demo1:向神经网络骨架中添加一个卷积层，并可视化查看卷积结果
"""
# 使用测试集，因为比较小
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)
    def forward(self, x):
        x = self.conv1(x)
        return x
tudui = Tudui()
print(tudui)
writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs, target = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    # torch.Size([64, 3, 32, 32])
    writer.add_images("before conv2d", imgs, step)
    # torch.Size([64, 6, 30, 30])
    # 因为channel是6，board不知道该怎么写入图片了，所以要reshape
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("after conv2d", output, step)
    step = step + 1
writer.close()
"""
demo2:向神经网络骨架中添加一个池化层，并可视化查看池化结果
这一部分代码和上边几乎一模一样，需要注意的是，池化层必须直接作用在float数据类型上，所以如果使用torch.tensor的话
就要加上dtype=float32，然后同样还要reshape为四维tensor
ceil mode：池化核走出input时还要不要里边的最大值 默认不要
"""
class Tudui2(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)
    def forward(self, input):
        output = self.maxpool1(input)
        return output
tudui = Tudui2()
writer = SummaryWriter("../logs_maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1
writer.close()
"""
demo3:向神经网络骨架中添加一个激活函数，并可视化结果
由于代码基本一致，只写网络骨架的代码吧
"""
class Tudui3(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()
    def forward(self, input):
        output = self.sigmoid1(input)
        return output
"""
demo4:向神经网络骨架中添加一个线性层，并可视化结果
由于代码基本一致，只写网络骨架的代码吧，只是特别注意这里在把图片放入线性层之前要用flatten把图片弄成一维的
"""
class Tudui4(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)
    def forward(self, input):
        output = self.linear1(input)
        return output
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # 线性层只能处理一维tensor
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui(output)
    print(output.shape)
"""
demo5:搭建一个vgg神经网络
"""
class Vgg(nn.Module):
    def __init__(self):
        super(Vgg, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x
vgg = Vgg()
input = torch.ones((64, 3, 32, 32))
output = vgg(input)
print(output.shape)
writer = SummaryWriter("logs")
writer.add_graph(vgg, input)
writer.close()
```

# 损失和优化器
```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
"""
demo1:在模型训练中加入损失函数和优化器
"""
dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x
# 这里注释是因为后边我保存/加载模型那一节的原因，在学习loss optim这一部分时可以取消注释
# tudui = Tudui()
# # 损失函数 优化器
# loss = nn.CrossEntropyLoss()
# optim = torch.optim.SGD(tudui.parameters(), lr=0.01)
#
# for i in range(20):
#     running_loss = 0.0
#     for data in dataloader:
#         imgs, targets = data
#         outputs = tudui(imgs)
#         result_loss = loss(outputs, targets)
#         # 注意 清零--》反向传播算梯度--》更新参数
#         optim.zero_grad()
#         result_loss.backward()
#         optim.step()
#         running_loss = running_loss + result_loss
#     print(running_loss)
# running_loss：
# tensor(18726.5977, grad_fn=<AddBackward0>)
# tensor(16132.8926, grad_fn=<AddBackward0>)
# tensor(15426.6357, grad_fn=<AddBackward0>)
```

# 现有网络模型的使用修改保存加载
```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
"""
demo1：加载vgg训练好的模型，并在里边加入一个线性层
"""
# ImageNet数据集太大了，100多G，还是用CIFAR10吧
# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                         transform=torchvision.transforms.ToTensor())
train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
# 加载现有的vgg模型
vgg16_not_pretrain = torchvision.models.vgg16(pretrained=False)
vgg16_pretrained = torchvision.models.vgg16(pretrained=True)
# 修改方法1：加入一个线性层,编号7
vgg16_pretrained.add_module("7", nn.Linear(1000, 10))
#vgg16_pretrained.classifier.add_module("7", nn.Linear(1000, 10))
print(vgg16_pretrained)
# 修改方法2：修改原来的第六个线性层
vgg16_not_pretrain.classifier[6] = nn.Linear(4096, 10)
print(vgg16_not_pretrain)
"""
demo2:保存/加载模型
两种保存方法 对应两种加载方法
保存模型都是用torch.save，加载模型都是用torch.load，一起保存的时候save整个模型，加载时直接torch.load加载
保存时只保存参数的，需要先向model vgg加载结构，再用model vgg.load state dict加载参数，加载参数还是要torc.load方法
保存方法1的‘陷阱’：
在使用方法1保存现有模型时，不会出错，代码更少，但是使用方法1保存自己的模型时，必须要引入这个模型的定义才可以
"""
# 保存东西需要现有东西保存
vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式1：模型结构+参数一起保存
torch.save(vgg16, "vgg16_pretrained_save_method1.pth")
# 多保存一个vgg16_pretrained，后面 完整模型测试讨论会用到
torch.save(vgg16_pretrained, "vgg16_pretrained_save_method1.pth")
# 加载方式1
model1 = torch.load("vgg16_save_method1.pth")
# 保存方式2：只保存模型参数
torch.save(vgg16.state_dict(), "vgg16_save_method2.pth")
# 加载方式2
# 先加载模型结构
model_vgg = torchvision.models.vgg16(pretrained=False)
# 再加载模型参数
model_vgg.load_state_dict(torch.load("vgg16_save_method2.pth"))
# 保存方法1的‘陷阱’
"""先保存tudui模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x
tudui = Tudui()
torch.save(tudui, "tudui_save_method1.pth")
"""
"""
直接加载tudui模型会报错
tudui = torch.load("tudui_save_method1.pth")
报错：
AttributeError: Can't get attribute 'Tudui' ...>
"""
"""
需要先把网络结构放进来，import 或者把class的定义代码粘贴过来
这里要把tu demo10中训练的代码注释掉，不然运行到class类之后，会继续往下执行训练的代码
虽然from代码没有高亮 但是如果删掉 程序就会报错
"""
from tu_demo10 import Tudui
tudui = torch.load("tudui_save_method1.pth")
print("Successfully load model!")
```

# 完整的模型训练套路
```python
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
"""
demo1:完整的模型训练测试流程
细节代码解释：
outputs.argmax(1)：outputs是一个元素是列表的列表，即二维矩阵，64行10列，argmax(1)是令行方向上的最大值为1，其余为0
with torch.no_grad():测试前要把梯度关闭
tudui.train() tudui.eval()：训练前 测试前写一下
"""
train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# 看一看训练集 测试集的长度
train_data_len = len(train_data)
test_data_len = len(test_data)
print("训练集的长度是：{}".format(train_data_len))
print("测试集的长度是：{}".format(test_data_len))
# 使用dataloader加载数据
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
# 创建网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32,5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self, x):
        x = self.model1(x)
        return x
# 创建模型
tudui = Tudui()
# 损失函数 优化器
loss_fn = nn.CrossEntropyLoss()
learing_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learing_rate)
# 设置控制训练次数的参数
# 记录训练 测试次数
total_train_step = 0
total_test_step = 0
# 训练轮数
epoch = 1
# 写入board
writer = SummaryWriter("logs")
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))
    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        # batch=64,训练集=5W，学习一边训练集就需要781.25次训练
        writer.add_scalar("train loss", loss.item(), total_train_step)
        if total_train_step % 100 == 0:
            print("训练次数：{}， loss：{}".format(total_test_step, loss.item()))
    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            print(targets)
            outputs = tudui(imgs)
            print(outputs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_len))
    writer.add_scalar("test loss", total_test_loss, total_test_step)
    writer.add_scalar("test accuracy", total_accuracy/test_data_len, total_test_step)
    total_test_step = total_test_step + 1
    # torch.save(tudui, "tudui_{}.pth".format(i))
    # print("模型已保存")
writer.close()
```

# 使用gpu
```python
"""
demo1:在特定位置加入.cuda()
能加的有3个地方：模型 loss 模型输入
    tudui = tudui.cuda()
    loss_fn = loss_fn.cuda()
    imgs, targets = data
    imgs = imgs.cuda()
    targets = targets.cuda()
    train_data没有cuda
    optimizer没有cuda
    
先判断再cuda
if torch,cuda_is_available():
    tudui = tudui.cuda()
    loss_fn = loss_fn.cuda()
    imgs, targets = data
    imgs = imgs.cuda()
    targets = targets.cuda()
"""
"""
demo2:在特定位置加入.to(device)
只有cpu
device = torch.device("cpu")
只有一张显卡
device = torch.device("cuda")
device = torch.device("cuda:0")
有多张显卡
device = torch.device("cuda:0")
device = torch.device("cuda:1")
能加的有3个地方：模型 loss 模型输入
    tudui = tudui.to(device)
    loss_fn = loss_fn.to(device)
    imgs, targets = data
    imgs = imgs.to(device)
    targets = targets.to(device)
先判断再to
device = torch.device("cuda", if torch.cuda_is_available() else "cpu")
"""
```

# 完整的模型验证
```python
import torch
import torchvision
from PIL import Image
"""
demo1:找一个图片让vgg model去识别是哪个类
代码解释：
image = image.convert('RGB')
由于图片有png jpg的不同格式，而png图片是四通道的 多一个透明度通道，jpg是三通道的 只有三个颜色通道
这一行代码可以让png jpg都只有三个颜色通道，增强了代码的适应性
demo14.py对应土堆的test.py 但是我这里加载的是vgg16 分类1000的模型 所以就不需要class的代码了
因为加class的代码是为了能按照加载方式1加载tudui模型
map_location=torch.device('cpu')
这个pth文件，如果是gpu训练得来的，就要放到gpu下运行，如果你想在cpu下运行，就加上这个代码
"""
# 加载图片
image_path = "./dataset/airplane.png"
image = Image.open(image_path)
image = image.convert('RGB')
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)
# 加载模型
model = torch.load("vgg16_pretrained_save_method1.pth", map_location=torch.device('cpu'))
print(model)
# 这里有个小疑问，明明模型是修改了的，加了线性层分类10类的，但是output还是1000个类别的概率
# 模型要四维带batch的图片类型，上边image.shape看到image是三维的
image = torch.reshape(image, (1, 3, 32, 32))
# 查看模型输出
model.eval()
with torch.no_grad():
    output = model(image)
print(output)
# 找到最大的概率值的位置 查看数字对应类别在debug datasets.CIFAR10的class to idx
print(output.argmax(1))
```