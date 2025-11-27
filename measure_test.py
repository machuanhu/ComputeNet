import requests
import random

if __name__ == '__main__':
    task_list = []
    task_list.append({
        'task_info':{
            'task_type':'common_task',
            'chip_info': 'GPU',
            'gpu_num':1,
        },
    })
    task_list.append({
        'task_info':{
            'task_type':'common_task',
            'chip_info': 'GPU',
            'gpu_num':1,
        },
    })
    task_list.append({
        'task_info':{
            'task_type':'common_task',
            'chip_info': 'GPU',
            'gpu_num':1,
        },
    })
    task_list.append({
        'task_info':{
            'task_type':'inferencing_task',
            'chip_info': 'GPU',
            'gpu_num':1,
            'content': '华为手机和小米手机哪个更好？'
        },
    })
    task_list.append({
        'task_info':{
            'task_type':'inferencing_task',
            'chip_info': 'GPU',
            'gpu_num':1,
            'content': '华为手机和小米手机哪个更好？'
        },
    })
    task_list.append({
        'task_info':{
            'task_type':'inferencing_task',
            'chip_info': 'GPU',
            'gpu_num':1,
            'content': '华为手机和小米手机哪个更好？'
        },
    })
    task_list.append({
        'task_info':{
            'task_type':'trainning_task',
            'chip_info': 'GPU',
            'gpu_num':1,
            'dataset': 'cifar',
            'model': 'resnet18',
            'loss_func': 'CrossEntropyLoss',
            'optimizer': 'Adam',
        },
    })
    task_list.append({
        'task_info':{
            'task_type':'trainning_task',
            'chip_info': 'GPU',
            'gpu_num':1,
            'dataset': 'cifar',
            'model': 'resnet18',
            'loss_func': 'CrossEntropyLoss',
            'optimizer': 'Adam',
        },
    })
    task_list.append({
        'task_info':{
            'task_type':'trainning_task',
            'chip_info': 'GPU',
            'gpu_num':1,
            'dataset': 'cifar',
            'model': 'resnet18',
            'loss_func': 'CrossEntropyLoss',
            'optimizer': 'Adam',
        },
    })
    selected_tasks = task_list
    print(selected_tasks)
    for task in selected_tasks:
        json_data = {
            'task':task
            }
        response = requests.post(f'http://127.0.0.1:60003/submit_task', json=json_data)
        result = response.json().get('result')