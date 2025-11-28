
from ns3test import Ns3Runner
import requests

from flask import Flask, request, jsonify

app = Flask(__name__)

def net_simulation(src_ip,dst_ip,body):
    semaphore = ns3_runner.add_request(src_ip,dst_ip,body)
    # print('wait for semaphore')
    semaphore.acquire()
    # print('semaphore ok')

@app.route('/rpc_call', methods=['POST'])
def rpc_call():
    json_data = request.get_json()
    src_ip = json_data['src_ip']
    dst_ip = json_data['dst_ip']
    method = json_data['method']
    params = json_data['params']

    body = {
        'method' : method,
        'params' : params,
    }
    self_ip="192.168.192.158"
    print(f'src_ip:{src_ip},dst_ip:{dst_ip},method:{method}')
    if dst_ip != self_ip:
        net_simulation(src_ip,dst_ip,body)

    params['caller_ip'] = src_ip
    try:
        if method=="task_completed" and dst_ip == self_ip:
            response = requests.post(f'http://{dst_ip}:60003/{method}', json=params, timeout=600)
        else:
            response = requests.post(f'http://{dst_ip}:60002/{method}', json=params, timeout=600)
    except Exception as e:
        print('Error forwarding:', e)
    
    return jsonify(response.json())

@app.route('/test', methods=['POST'])
def test():
    json_data = request.get_json()
    src_ip = json_data['src_ip']
    dst_ip = json_data['dst_ip']
    method = json_data['method']
    params = json_data['params']

    body = {
        'method' : method,
        'params' : params,
    }

    net_simulation(src_ip,dst_ip,body)
    
    response = {
        'result': True
    }
    return jsonify(response)

if __name__ == '__main__':
    ns3_runner = Ns3Runner()
    ns3_runner.run()

    app.run(host='0.0.0.0', port=60001)