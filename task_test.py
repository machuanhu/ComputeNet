import requests
import random

if __name__ == '__main__':
    task_list = []
    task_list.append({
        'task_info':{
            'task_type':'inferencing_task',
            'chip_info': 'GPU',
            'gpu_num':1,
            'content': '天空为什么是蓝色的？'
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
            'content': '火车径直前行会压死五个被绑在铁轨上的人，但拉下拉杆会变道压死另一个人，你会怎么做？'
        },
    })
    # task_list.append({
    #     'task_info':{
    #         'task_type':'trainning_task',
    #         'chip_info': 'GPU',
    #         'gpu_num':1,
    #         'dataset': 'Mnist',
    #         'model': 'resnet18',
    #         'loss_func': 'trainning_task',
    #         'optimizer': 'Adam'
    #     },
    # })
    task_list.append({
        'task_info':{
            'task_type':'trainning_task',
            'chip_info': 'CPU',
            'gpu_num':0,
            'dataset': 'Mnist',
            'model': 'resnet18',
            'loss_func': 'CrossEntropyLoss',
            'optimizer': 'Adam'
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
            'task_type':'common_task',
            'chip_info': 'CPU',
            'gpu_num':0,
        },
    })

    selected_tasks = random.choices(task_list, k=20)
    print(selected_tasks)
    for task in selected_tasks:
        json_data = {
            'task':task
            }
        response = requests.post(f'http://127.0.0.1:60002/submit_task', json=json_data)
        result = response.json().get('result')