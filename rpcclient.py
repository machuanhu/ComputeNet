import xmlrpc.client
import socket
import requests
import resources
import json
import pickle

ns3_address = '192.168.192.158:60001'
src_ip = '192.168.192.158'
dst_ip = '192.168.192.175'
data = 'what can I say'

json_data = {
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'method': 'test',
        'params': {}
        }
response = requests.post(f'http://{ns3_address}/test', json=json_data)
result = response.json().get('result')