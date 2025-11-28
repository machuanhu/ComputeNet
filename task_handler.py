import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST,CIFAR10
import torchvision.transforms as transforms
from torchvision.models import resnet18
import torch.nn.functional as F
import numpy as np
import time
import measure_rpcserver
device=measure_rpcserver.device
device_=measure_rpcserver.device_



def handle_task(task):
    if task['task_info']['task_type'] == 'common_task':
        return handle_common_task(task)
    elif task['task_info']['task_type'] == 'trainning_task':
        return handle_trainning_task(task)
    elif task['task_info']['task_type'] == 'inferencing_task':
        return handle_inferencing_task(task)

def handle_common_task(task):
    start_time = time.time()
    if task['task_info']['chip_info'] == 'GPU':
        output = common_task1()
    else:
        output = common_task2()
    end_time = time.time()
    duration = end_time - start_time

    done_task = {
        'uuid': task['uuid'],
        'task_start_time': start_time,
        'task_end_time': end_time,
        'task_duration': duration,
        'task_type' : "common_task",
        'result': output
    }

    return done_task

def handle_trainning_task(task):
    task_info = task['task_info']
    start_time = time.time()
    mnist_task(task_info['gpu_num'])
    end_time = time.time()
    duration = end_time - start_time

    done_task = {
        'uuid': task['uuid'],
        'task_start_time': start_time,
        'task_end_time': end_time,
        'task_duration': duration,
        'task_type' : "train_task",
        'result': 'trained_model'
        }
    return done_task

def handle_inferencing_task(task):
    task_info = task['task_info']
    content = task_info['content']

    start_time = time.time()
    output = inference(content)
    end_time = time.time()
    duration = end_time - start_time

    done_task = {
        'uuid': task['uuid'],
        'task_start_time': start_time,
        'task_end_time': end_time,
        'task_duration': duration,
        'task_type' : "inference_task",
        'result': output
    }

    return done_task

################################################################################

def common_task1():
    # 定义矩阵的大小
    matrix_size = 1000
    # 生成两个随机矩阵
    a = torch.rand(matrix_size, matrix_size, device=device)
    b = torch.rand(matrix_size, matrix_size, device=device)

    # 执行矩阵相乘并计算时间
    start_time = time.time()
    torch.mm(a, b)
    end_time = time.time()

    # 计算运行时间
    execution_time = end_time - start_time
    print(f"Matrix multiplication of size {matrix_size}x{matrix_size} took {execution_time} seconds")

def common_task2():
        # 定义矩阵的大小
    matrix_size = 1000

    # 生成两个随机矩阵
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    # 执行矩阵相乘并计算时间
    start_time = time.time()
    result = np.dot(matrix_a, matrix_b)
    end_time = time.time()

    # 计算运行时间
    execution_time = end_time - start_time
    print(f"Matrix multiplication of size {matrix_size}x{matrix_size} took {execution_time} seconds")

################################################################################

def inference(content,model='llama3.1'):
    model = MnistNet().to(device_)
    dummy_input = torch.randn(64, 1, 28, 28, device=device_)
    model.eval()
    with torch.no_grad():
        for _ in range(2):
            _ = model(dummy_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        with torch.no_grad():
            for _ in range(2):
                _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 1000
        avg_time = elapsed_ms / 2
    else:
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = model(dummy_input)
        avg_time = (time.perf_counter() - start) / 2
    del model, dummy_input
    torch.cuda.empty_cache()
    return f"inference_completed in {avg_time:.4f}s (avg of 100 runs)"

################################################################################

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)

        return x
    
class CifarNet(nn.Module):
    def __init__(self,in_channels):
        super(CifarNet, self).__init__()
        self.resnet18 = resnet18(pretrained=False, num_classes=10)
        self.resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        x = self.resnet18(x)
        return x

def mnist_task(gpu_num):
    use_gpu = True if gpu_num >= 1 else False
    
    model = MnistNet().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=16,drop_last=True,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/5], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')
                # time.sleep(0.1/rate)

def cifar_task(gpu_num):
    use_gpu = True if gpu_num >= 1 else False
    
    model = CifarNet(3).to(device)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=16,drop_last=True,shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch [{epoch+1}/5], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')
                # time.sleep(0.1/rate)

# def train(dataset,model,loss_func,optimizer,gpu_num,rate):
#     use_gpu = True if gpu_num >= 1 else False
#     device = torch.device('cuda:0' if use_gpu else 'cpu')
    
#     if dataset == 'Mnist':
#         model = MnistNet().to(device)
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,)),
#         ])
#         train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
#     elif dataset == 'cifar':
#         model = CifarNet(3).to(device)
#         transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
#         train_dataset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
        
#     train_loader = DataLoader(train_dataset, batch_size=64,drop_last=True,shuffle=True)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters())
    
#     for epoch in range(5):
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(device), target.to(device)

#             optimizer.zero_grad()
#             output = model(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()

#             if batch_idx % 10 == 0:
#                 print(f'Epoch [{epoch+1}/5], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}')
#                 time.sleep(0.1/rate)

if __name__ == '__main__':
    content = '天空为什么是蓝色的'
    output = inference(content)
    print(output)