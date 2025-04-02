import resources
import pickle
import json
import requests
import argparse
import json
import time
import multiprocessing as mp

class Identification:
    def __init__(self,location,industry,ownership,resource_type,node_info,chip_info,area,other_info):
        self.location = location
        self.industry = industry
        self.ownership = ownership
        self.resource_type = resource_type
        self.node_info = node_info
        self.chip_info = chip_info
        self.area = area
        self.other_info = other_info
        
    def show_attributes(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

    def to_string(self):
        return '.'.join([f"{value}" for key, value in vars(self).items()])

class Node:
    def __init__(self,id,idt):
        self.id = id
        self.idt = idt
        self.address = None

def get_identifications():
    file_path = "identifications.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)

    idt_list = []
    for idt_data in data:
        idt = Identification(**idt_data)
        idt_list.append(idt)
    return idt_list
    

def get_address_table():
    # idt_list = get_identifications()
    # idt_to_address = {}
    # idt_to_address[idt_list[0].to_string()] = '192.168.124.103'
    # idt_to_address[idt_list[1].to_string()] = '192.168.124.104'
    # idt_to_address[idt_list[2].to_string()] = '192.168.124.105'
    # print(idt_to_address)

    # hash_file_path = "address_mapping.json"
    # with open(hash_file_path, "w") as json_file:
    #     json.dump(idt_to_address, json_file, indent=4)

    file_path = "address_mapping.json"
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data
        
def get_sn(idt):
    json_data = {
        'src_ip': src_ip,
        'dst_ip': node.address,
        'method': 'req_sn',
        'params': {}
        }
    response = requests.post(f'http://{ns3_address}/rpc_call', json=json_data)
    result = response.json().get('result')
    sn = pickle.loads(json.loads(result).encode('latin1'))
    return sn
        
def measure(sn_list):
    weights = [0.4,0.3,0.2,0.1]
    evaluator = resources.Evaluator(sn_list,weights)
    compute_scores,communication_scores,memory_scores,storage_scores,overall_scores = evaluator.evaluate()

    for i,node in enumerate(node_list):
        json_data = {
                'src_ip': src_ip,
                'dst_ip': node.address,
                'method': 'dynamic_evaluate',
                'params': {}
                }
        response = requests.post(f'http://{ns3_address}/rpc_call', json=json_data)
        result = response.json().get('result')
        sn_list[i].update_dynamic_indicators(**result)
    
    sorted_indexes = sorted(range(len(sn_list)), key=lambda i: sn_list[i].dynamic_score, reverse=True)
    print('sorted_indexes:',sorted_indexes)
    return sorted_indexes

def dispatch_tasks(seq=(0,1,2)):
    task_info_list = []
    task_info_list.append({
        'task_id': 0,
        'dataset': 'Mnist',
        'model': 'resnet18',
        'loss_func': 'CrossEntropyLoss',
        'optimizer': 'Adam',
        'chip_info': 'CPU',
        'gpu_num':0,
        })
    task_info_list.append({
        'task_id': 1,
        'dataset': 'cifar',
        'model': 'resnet18',
        'loss_func': 'CrossEntropyLoss',
        'optimizer': 'Adam',
        'chip_info': 'GPU',
        'gpu_num':1,
        })
    task_info_list.append({
        'task_id': 2,
        'dataset': 'Mnist',
        'model': 'resnet18',
        'loss_func': 'CrossEntropyLoss',
        'optimizer': 'Adam',
        'chip_info': 'GPU',
        'gpu_num':1,
        })
    for i,node in enumerate(node_list):
        print(task_info_list[seq[i]])
        json_data = {
                'src_ip': src_ip,
                'dst_ip': node.address,
                'method': 'dispatch_task',
                'params': {
                    'task_info':task_info_list[seq[i]]
                    }
                }
        response = requests.post(f'http://{ns3_address}/rpc_call', json=json_data)
        result = response.json().get('result')
        print(f'task{seq[i]} dispatched')

def dispatch_inference_tasks(seq=(0,1,2)):
    task_info_list = []
    task_info_list.append({
        'task_id': 0,
        'content': '天空为什么是蓝色的？',
    })
    task_info_list.append({
        'task_id': 1,
        'content': '华为手机和小米手机哪个更好？',
    })
    task_info_list.append({
        'task_id': 2,
        'content': '火车径直前行会压死五个被绑在铁轨上的人，但拉下拉杆会变道压死另一个人，你会怎么做？',
    })

    process_list = []
    for i,task_info in enumerate(task_info_list):
        print(task_info)
        json_data = {
                'src_ip': src_ip,
                'dst_ip': node_list[seq[i]].address,
                'method': 'ollama_test',
                'params': {
                    'task_info':task_info
                    }
                }
        process = mp.Process(target=run_task,args=(json_data,))
        process.start()
        process_list.append(process)
        print(f'task{seq[i]} dispatched')

    for process in process_list:
        process.join()

def run_task(data):
    response = requests.post(f'http://{ns3_address}/rpc_call', json=data)
    recv_time = time.time()
    response_data = response.json()
    start_time = response_data['start_time']
    end_time = response_data['end_time']
    result = response_data['result']
    print(result)

    duration = end_time - start_time

    log_data = {
        "task_id": data['params']['task_info']['task_id'],
        "taker_ip": data['dst_ip'],
        "start_time": start_time,
        "end_time": end_time, 
        "recv_time": recv_time, 
        "duration": duration, 
        "result": result,
        }
    with open("done_task_log.txt", 'a') as file_object:
        file_object.write(json.dumps(log_data,ensure_ascii=False)+'\n')

if __name__ == '__main__':
    # python measure_test.py --use_measure
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_measure', action='store_true', help='use measure or not')
    args = parser.parse_args()
    ns3_address = '192.168.124.101:60001'
    src_ip = '192.168.124.102'

    idt_list =  get_identifications()
    node_list = []
    for i,idt in enumerate(idt_list):
        node = Node(id=i,idt=idt)
        node_list.append(node)

    address_table = get_address_table()
    for i,node in enumerate(node_list):
        print(f'identity{i+1}')
        node.idt.show_attributes()
        node.address = address_table[node.idt.to_string()]
        print(f'mapping identity{i+1} to IP address {node.address}')
    
    if args.use_measure:
        sn_list = []
        for node in node_list:
            sn_list.append(get_sn(node))
        sorted_indexes = measure(sn_list)
        dispatch_inference_tasks(sorted_indexes)
    else:
        dispatch_inference_tasks((0,1,2))