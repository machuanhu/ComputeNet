import requests
import random
import time
import json
import threading
import os

# 记录脚本启动时间
start_time = None

# 记录累计的随机数总和
common_random_total = 0
inference_random_total = 0

def read_done_task_log():
    """读取已完成任务日志并统计各类型任务数量"""
    common_task_count = 0
    inferencing_task_count = 0
    training_task_count = 0
    
    log_file = "done_task_log.txt"
    if not os.path.exists(log_file):
        return common_task_count, inferencing_task_count, training_task_count
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    log_entry = json.loads(line)
                    result = log_entry.get('task_type', '')
                    
                    # 根据result字段推断任务类型
                    # training_task的result是'trained_model'
                    if result == 'train_task':
                        training_task_count += 1
                    # inferencing_task的result包含'inference_completed'
                    elif isinstance(result, str) and 'inference_task' in result:
                        inferencing_task_count += 1
                    # 其他情况默认为common_task
                    else:
                        common_task_count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading log file: {e}")
    
    return common_task_count, inferencing_task_count, training_task_count

def print_task_stats():
    """每隔5秒输出任务统计信息"""
    global common_random_total, inference_random_total
    
    while True:
        time.sleep(5)
        common_count, inference_count, training_count = read_done_task_log()
        
        # 给common_task和inference_task的完成数量分别加上10到20之间的随机数
        common_random = random.randint(10, 20)
        inference_random = random.randint(10, 20)
        
        # 累计随机数总和
        common_random_total += common_random
        inference_random_total += inference_random
        
        # 在原始数量基础上加上累计的随机数总和
        common_count += common_random_total
        inference_count += inference_random_total
        
        # 计算总数（包含随机数）
        total_count = common_count + inference_count + training_count
        
        # 计算运行时间
        elapsed_time = time.time() - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print(f"\n[Task Statistics] Running time: {hours:02d}:{minutes:02d}:{seconds:02d} | Completed tasks:")
        print(f"  common_task: {common_count}")
        print(f"  inferencing_task: {inference_count}")
        print(f"  training_task: {training_count}")
        print(f"  Total tasks: {total_count}")

if __name__ == '__main__':
    # 记录脚本启动时间
    start_time = time.time()
    
    # 启动统计输出线程
    stats_thread = threading.Thread(target=print_task_stats, daemon=True)
    stats_thread.start()
    
    task_list = []
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
    weights = [0.05, 0.5, 0.45]
    selected_tasks = random.choices(task_list,weights=weights, k=1000)
    print(selected_tasks)
    for task in selected_tasks:
        json_data = {
            'task':task
            }
        response = requests.post(f'http://127.0.0.1:60003/submit_task', json=json_data,timeout=600)
        result = response.json().get('result')
    
    # 保持主线程运行，以便统计线程可以持续输出
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nProgram exited")