from queue import Queue
from threading import Thread
import socket
import requests
import time
import json
import pickle
import uuid
import resources
from identifications_mapping import Mapper, get_identifications

class Scheduler:
    def __init__(self, ip, ns3_address) -> None:
        self.ip = ip
        self.ns3_address = ns3_address
        self.task_queue = Queue()
        self.task_counter = 0
        self.running_tasks = {}

    def run(self):
        self.scheduler_thread = Thread(target=self.main)
        self.scheduler_thread.start()

    def main(self):
        idt_list = get_identifications()
        mapper = Mapper()
        node_list = []
        task_count=1
        sorted_index = 0
        for i,idt in enumerate(idt_list):
            print(f'identity{i+1}')
            idt.show_attributes()
            address = mapper.get_address(idt)
            print(f'mapping identity{i+1} to IP address {address}')

            node = resources.NodeContainer(
                id=i,
                idt=idt,
                address=address,
                sn=get_sn(self.ip,address,self.ns3_address)
                )
            node_list.append(node)

        evaluator = resources.Evaluator(
            service_nodes=[node.sn for node in node_list],
            general_weights=[0.4,0.3,0.2,0.1],
            special_weights=[0.3,0.5,0.1,0.1]
            )
        evaluator.evaluate()

        while(task := self.task_queue.get()):
            print('!!!!!!!!!!!',task)
            task_send_ok = False
            while not task_send_ok:
                if task_count % 100 == 1:
                    for node in node_list:
                        json_data = {
                                'src_ip': self.ip,
                                'dst_ip': node.address,
                                'method': 'dynamic_evaluate',
                                'params': {}
                                }
                        response = requests.post(f'http://{self.ns3_address}/rpc_call', json=json_data)
                        result = response.json().get('result')
                        print(result)
                        node.sn.update_dynamic_indicators(**result)
                        sorted_index=0

                selected_node_list = select_node(node_list,task,evaluator)
                selected_node = selected_node_list[sorted_index%len(selected_node_list)]
                if selected_node is not None:
                    task_count+=1
                    sorted_index+=1
                    send_task_to_node(selected_node,task,self.ip,self.ns3_address,self.running_tasks)
                    task_send_ok = True
                
                #time.sleep(5)

    def add_task(self,task):
        task['task_id'] = self.task_counter
        task['task_submission_time'] = time.time()
        self.task_queue.put(task)
        self.task_counter += 1

    def task_done(self,unique_id):
        task,node,using_chip = self.running_tasks.pop(unique_id)
        if using_chip == 'CPU':
            node.using_cpu = False
        elif using_chip == 'GPU':
            node.using_gpu = False
        return task

def get_sn(src_ip,address,ns3_address):
    json_data = {
        'src_ip': src_ip,
        'dst_ip': address,
        'method': 'req_sn',
        'params': {}
        }
    response = requests.post(f'http://{ns3_address}/rpc_call', json=json_data)
    result = response.json().get('result')
    sn = pickle.loads(json.loads(result).encode('latin1'))
    return sn

def select_node(node_list,task,evaluator):
    if task['task_info']['task_type'] == 'common_task':
        dynamic_scores = evaluator.get_dynamic_scores(is_general=True)
    else:
        dynamic_scores = evaluator.get_dynamic_scores(is_general=False)
    
    available_node_list = []
    if task['task_info']['chip_info'] == 'CPU':
        available_node_list = [node for node in node_list]
    elif task['task_info']['chip_info'] == 'GPU':
        available_node_list = [node for node in node_list]
    if len(available_node_list) <= 0:
        return None
    else:
        sorted_list = sorted(available_node_list, key=lambda node: dynamic_scores[node_list.index(node)], reverse=True)
        return sorted_list

def send_task_to_node(node,task,src_ip,ns3_address,running_tasks):
    unique_id = uuid.uuid4()
    uuid_str = str(unique_id)
    json_data = {
        'src_ip': src_ip,
        'dst_ip': node.address,
        'method': 'dispatch_task',
        'params': {
            'task':{
                'uuid':uuid_str,
                'task_info':task['task_info'],
                }
            }
        }
    response = requests.post(f'http://{ns3_address}/rpc_call', json=json_data)
    result = response.json().get('result')

    using_chip = ''
    if task['task_info']['chip_info'] == 'CPU':
        node.using_cpu = True
        using_chip = 'CPU'
    elif task['task_info']['chip_info'] == 'GPU':
        node.using_gpu = True
        using_chip = 'GPU'
    running_tasks[uuid_str] = [task,node,using_chip]

if __name__ == '__main__':
    ns3_address = '192.168.192.158:60001'
    scheduler = Scheduler(ns3_address)
    scheduler.run()