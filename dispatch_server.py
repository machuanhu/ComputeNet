from flask import Flask, request, jsonify
from schedule import Scheduler
import argparse
import time
import json

app = Flask(__name__)

@app.route('/submit_task', methods=['POST'])
def submit_task():
    json_data = request.get_json()
    task = json_data['task']
    scheduler.add_task(task)

    response = {
        'result': True
    }
    return jsonify(response)      

@app.route('/task_completed', methods=['POST'])
def task_completed():
    json_data = request.get_json()
    caller_ip = json_data['caller_ip']
    done_task = json_data['done_task']
    
    print('caller_ip:',caller_ip)
    print('done_task:',done_task)
    
    unique_id = done_task['uuid']
    task = scheduler.task_done(unique_id)

    done_task_log = {}
    done_task_log['task_id'] = task['task_id']
    done_task_log['task_submission_time'] = task['task_submission_time']
    done_task_log['task_completion_time'] = time.time()
    done_task_log['task_duration'] = done_task['task_duration']
    done_task_log['taker_ip'] = caller_ip
    done_task_log['result'] = done_task['result']
    with open("done_task_log.txt", 'a', encoding='utf-8') as file_object:
        file_object.write(json.dumps(done_task_log,ensure_ascii=False)+'\n')
    
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
    parser.add_argument('--ns3_address', default='192.168.124.101:60001', type=str, help='ns3_address')
    args = parser.parse_args()
    scheduler = Scheduler(ns3_address=args.ns3_address)
    scheduler.run()
    app.run(host='0.0.0.0', port=60002)