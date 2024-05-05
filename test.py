"""
Revelations
1.Data preprocessing is important
2.Complex net need to solve overfitting
3.The width and the depth of net both matters
"""
import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
from CNN import CNN

train_dataset = CIFAR10('./CIFAR10', train=True, download=False, transform=transforms.transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
     transforms.RandomHorizontalFlip(),  # 随机水平镜像
     transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
     transforms.RandomCrop(32, padding=4),
     ]))
test_dataset = CIFAR10('./CIFAR10', train=False, download=False, transform=transforms.transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda:0")

print('Loading the model...')
model = CNN()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      nesterov=True,
                      weight_decay=5e-4
                      )

losses = []
acces = []
eval_losses = []
eval_acces = []

for epoch in range(1, 101):
    train_loss = 0
    train_acc = 0
    model.train()
    # 动态修改参数学习率
    if epoch == 10:
        optimizer.param_groups[0]['lr'] *= 0.1

    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)

        img = img.view(img.size(0), 3, 32, 32)

        # 前向传播
        out = model(img)
        loss = criterion(out, label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录误差
        train_loss += loss.item()

        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc

    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))

    # 在测试集上检测效果
    eval_loss = 0
    eval_acc = 0

    # 将模型改为预测模式
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        img = img.view(img.size(0), 3, 32, 32)

        out = model(img)
        loss = criterion(out, label)

        # 记录误差
        eval_loss += loss.item()

        # 记录准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print(
        'epoch: {}, learn rate: {:.4f}, Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(
            epoch, optimizer.param_groups[0]['lr'],
            train_loss / len(train_loader), train_acc / len(train_loader),
            eval_loss / len(test_loader),
            eval_acc / len(test_loader)
        ))
