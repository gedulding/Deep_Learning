import torch;
from torch import  nn;
from torch.nn import functional as F;
from torch import  optim;

import torchvision
import matplotlib.pyplot as plt;
from utils import plot_curve,plot_image,one_hot;

batch_size = 512;
#加载数据
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))         #手写数字只有黑白，所以只有01两种像素，所以需要将数据进行正态化，可以提升准确率
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x , y = next(iter(train_loader));
print(x.shape , y.shape);

class Net(nn.Module):
    def __init__(self):
        super(Net , self).__init__();
        self.fc1 = nn.Linear(28*28 , 256);
        self.fc2 = nn.Linear(256,64);
        self.fc3 = nn.Linear(64 , 10);

    #前向传播计算
    def forward(self , x):
        # w1*x + b1
        x = F.relu(self.fc1(x));
        x = F.relu(self.fc2(x));

        x = self.fc3(x);
        return x;

net  = Net();
optimizer = optim.SGD(net.parameters() , lr = 0.01 , momentum=0.9);
train_loss = [];
for epoch in range(3):
    for batch_idx , (x , y) in enumerate(train_loader):
        x = x.view(x.size(0) , 1 * 28 * 28);
        out  = net(x);
        y_onehot = one_hot(y);
        #损失函数
        loss  = F.mse_loss(out , y_onehot);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        train_loss.append(loss.item());
        if batch_idx % 10 == 0:
            print(epoch , batch_idx , loss.item());

#绘制损失曲线
plot_curve(train_loss);

total_correct = 0;
for x , y in test_loader:
    x = x.view(x.size(0) , 1 * 28 * 28);
    out = net(x);
    pred = out.argmax(dim = 1);
    correct = pred.eq(y).sum().float();
    total_correct += correct;

total_num = len(test_loader.dataset);
acc = total_correct / total_num;
print("准确率：" , acc);

#可视化比对
x , y = next(iter(test_loader));     #取一个batch
out = net(x = x.view(x.size(0) , 1 * 28 * 28));
pred = out.argmax(dim = 1);
plot_image(x , pred , 'test');



