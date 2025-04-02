import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet18 = resnet18(pretrained=False, num_classes=10)
        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 修改输入通道数为1

    def forward(self, x):
        x = self.resnet18(x)
        return x

def train(rank, world_size, host, port):
    # 设置分布式训练环境
    dist.init_process_group(backend='nccl', init_method=f'tcp://{host}:{port}', rank=rank, world_size=world_size)

    # 分配 GPU 设备
    torch.cuda.set_device(0)
    model = Net().cuda()
    model = nn.parallel.DistributedDataParallel(model,device_ids=[0],output_device=0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # 加载 MNIST 数据集
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)

    # 创建模型和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(5):
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Rank {rank}: Epoch [{epoch+1}/5], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')

    # 释放分布式训练资源
    dist.destroy_process_group()

if __name__ == '__main__':
    # python dist_test.py --ps_ip=192.168.124.101 --ps_port=23456 --use_gpu --world_size=2 --rank=0
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--ps_ip', default='localhost', type=str, help='ip of ps')
    parser.add_argument('--ps_port', default='8888', type=str, help='port of ps')
    parser.add_argument('--use_gpu', action='store_true', help='use gpu or not')
    args = parser.parse_args()

    args.world_size = 3
    args.rank = 0
    args.ps_ip = '192.168.124.101'
    args.ps_port = 12345

    mp.spawn(train, args=(args.world_size, args.ps_ip, args.ps_port), nprocs=1, join=True)