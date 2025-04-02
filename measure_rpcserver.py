import argparse
import pickle
import resources
import json
import torch.multiprocessing as mp
import time
import requests
import socket
import task_handler

from flask import Flask, request, jsonify

app = Flask(__name__)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def init(id):
    sn_list = []
    sn_MLU = resources.ServiceNode("MLU",
                                [2000.85,212.64,1416797],   #整数运算速率，浮点运算速率，哈希运算速率
                                [100],  ##带宽Mb/s
                                [1024,11790.8], #内存容量GB,内存带宽MB/s
                                [12480,21.8,5312]   #存储容量GB,存储带宽MB/s,每秒读写次数ops/s
                                )
    
    sn_GCU = resources.ServiceNode("GCU",
                                [1995.96,358.68,2402351],
                                [100],
                                [384,16131.5],
                                [17280,466,124700]
                                )
    
    sn_4090 = resources.ServiceNode("4090",
                                [28066.47,675.32,1735655],
                                [200],
                                [125,17208.4],
                                [8400,491.9,139700]
                                )
    sn_tyy = resources.ServiceNode("tyy",
                                [10000,400,1500000],
                                [500],
                                [512,12000],
                                [5120,500,30000]
                                )
    sn_NPU = resources.ServiceNode("NPU",
                                [5000,500,100000],
                                [300],
                                [11,11000],
                                [256,30,10000]
                                )
    sn_list.append(sn_MLU)
    sn_list.append(sn_GCU)
    sn_list.append(sn_4090)
    sn_list.append(sn_tyy)
    sn_list.append(sn_NPU)

    ip = get_local_ip()

    return sn_list[id],ip

def run_task(task):
    done_task = task_handler.handle_task(task)
    
    caller_ip = task['caller_ip']
    json_data = {
        'src_ip': ip,
        'dst_ip': caller_ip,
        'method': 'task_completed',
        'params': {
            'done_task': done_task
            }
        }
    response = requests.post(f'http://{ns3_address}/rpc_call', json=json_data)
    result = response.json().get('result')

def start_task(task):
    process = mp.Process(target=run_task,args=(task,))
    process.start()

@app.route('/req_sn', methods=['POST'])
def req_sn():
    json_data = request.get_json()
    caller_ip = json_data['caller_ip']
    print('caller_ip:',caller_ip)
    
    response = {
        'result': json.dumps(pickle.dumps(sn).decode('latin1'))
    }
    return jsonify(response)

@app.route('/dynamic_evaluate', methods=['POST'])
def dynamic_evaluate():
    json_data = request.get_json()
    caller_ip = json_data['caller_ip']
    print('caller_ip:',caller_ip)

    response = {
        'result': sn.dynamic_metrics()
    }
    return jsonify(response)

@app.route('/dispatch_task', methods=['POST'])
def dispatch_task():
    json_data = request.get_json()
    caller_ip = json_data['caller_ip']
    task = json_data['task']

    print('caller_ip:',caller_ip)
    print('task:',task)

    task['caller_ip'] = caller_ip
    start_task(task)

    response = {
        'result': True
    }
    return jsonify(response)

@app.route('/test', methods=['POST'])
def test():
    json_data = request.get_json()
    caller_ip = json_data['caller_ip']
    data = json_data['data']

    print('caller_ip:',caller_ip)
    print('data:',data)

    response = {
        'result': True
    }
    return jsonify(response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default=0, type=int, help='id')
    parser.add_argument('--ns3_address', default='192.168.124.101:60001', type=str, help='ns3_address')
    args = parser.parse_args()
    ns3_address = args.ns3_address
    sn,ip = init(args.id)
    print(ip)

    app.run(host='0.0.0.0', port=60002)