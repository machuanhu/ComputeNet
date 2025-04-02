import numpy as np
import psutil
import platform
import subprocess
import time

# class ComputeResource:
#     def __init__(self, name, frequency, float_calculation_rate, int_calculation_rate):
#         self.name = name
#         self.frequency = frequency
#         self.float_calculation_rate = float_calculation_rate
#         self.int_calculation_rate = int_calculation_rate

#         self.score = 0
    
#     def get_performance_indicators(self):
#         return np.array([self.frequency,self.float_calculation_rate,self.int_calculation_rate])
    
#     def set_score(self,score):
#         self.score = score

#     def get_score(self):
#         return self.score

# class CommunicationResource:
#     def __init__(self, name, network_bandwidth):
#         self.name = name
#         self.network_bandwidth = network_bandwidth

#         self.score = 0

#     def get_performance_indicators(self):
#         return np.array([self.network_bandwidth])
    
#     def set_score(self,score):
#         self.score = score

#     def get_score(self):
#         return self.score

# class MemoryResource:
#     def __init__(self, name, memory_capacity, memory_bandwidth):
#         self.name = name
#         self.memory_capacity = memory_capacity
#         self.memory_bandwidth = memory_bandwidth

#         self.score = 0

#     def get_performance_indicators(self):
#         return np.array([self.memory_capacity,self.memory_bandwidth])
    
#     def set_score(self,score):
#         self.score = score

#     def get_score(self):
#         return self.score

# class StorageResource:
#     def __init__(self, name, storage_apacity, storage_bandwidth, iops):
#         self.name = name
#         self.storage_apacity = storage_apacity
#         self.storage_bandwidth = storage_bandwidth
#         self.iops = iops

#         self.score = 0

#     def get_performance_indicators(self):
#         return np.array([self.storage_apacity,self.storage_bandwidth,self.iops])
    
#     def set_score(self,score):
#         self.score = score

#     def get_score(self):
#         return self.score
    
class ServiceNode:
    def __init__(self, name, compute_indicators, communication_indicators, memory_indicators, storage_indicators):
        self.name = name

        self.compute_indicators = compute_indicators
        self.communication_indicators = communication_indicators
        self.memory_indicators = memory_indicators
        self.storage_indicators = storage_indicators

        self.compute_score = 0
        self.communication_score = 0
        self.memory_score = 0
        self.storage_score = 0
        self.overall_score = 0
        
        self.busyness = 0
        self.dynamic_score = 0

        self.gpu_info = None
        self.cpu_info = None
        self.net_info = None
        self.mem_info = None
        self.storage_info = None
        
        # self.current_sessions_num = 0
        # self.accessed_sessions_count = 0
        # self.sessions_info = None
    
    # def access_session(self):
    #     access_time = time.time()
    #     estimated_duration = access_time + 10
    #     occupied_gpu_id = 0
    #     self.gpu_info[occupied_gpu_id]['occupied'] = True
    #     self.current_sessions_num += 1
    #     self.accessed_sessions_count += 1
    #     session_id = self.accessed_sessions_count
    #     self.sessions_info[session_id] = {
    #         'session_id':session_id,
    #         'access_time':access_time,
    #         'estimated_duration':estimated_duration,
    #         'occupied_gpu_id':occupied_gpu_id
    #     }
    #     self.update_dynamic_indicators(**self.dynamic_metrics())

    # def end_session(self,session_id):
    #     ended_session_info = self.sessions_info.pop(session_id)
    #     self.gpu_info[0][ended_session_info['occupied_gpu_id']] = False
    #     self.current_sessions_num -= 1
    #     self.update_dynamic_indicators(**self.dynamic_metrics())

    def get_cpu_info(self):
        cpu_count = psutil.cpu_count(logical=False)
        sys_cpu_usage = psutil.cpu_percent(interval=1)

        cpu_info = {
            'cpu_count':cpu_count,
            'cpu_usage':sys_cpu_usage
        }
        
        return cpu_info

    def get_gpu_info(self):
        gpu_info = {}
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
            output = result.stdout.strip()
            gpu_models = output.split('\n')

            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True)
            output = result.stdout.strip()
            gpu_count = len(output.split('\n'))
            
            for i in range(gpu_count):
                result = subprocess.run(['nvidia-smi', '--id=' + str(i), '--query-gpu=utilization.gpu', '--format=csv,noheader'], capture_output=True, text=True)
                output = result.stdout.strip()
                usage = int(output.split()[0])
                gpu_info[str(i)] = {
                    'gpu_id':i,
                    'gpu_model':gpu_models[i],
                    'gpu_usage':usage,
                    'occupied':self.gpu_info and self.gpu_info[i]['occupied'] or False
                }
        except:
            print('no gpu!')
        
        return gpu_info

    def get_mem_info(self):
        memory_info = psutil.virtual_memory()
        mem_usage = memory_info.percent

        total_memory_bytes = memory_info.total
        used_memory_bytes = memory_info.used
        free_memory_bytes = memory_info.available

        mem_info = {
            'total_memory_bytes':total_memory_bytes,
            'used_memory_bytes':used_memory_bytes,
            'free_memory_bytes':free_memory_bytes,
            'mem_usage':mem_usage
        }

        return mem_info

    def get_storage_info(self):
        partitions = psutil.disk_partitions(all=False)
        storage_total = 0
        storage_used = 0
        storage_free = 0
        storage_usage = 0

        for partition in partitions:
            disk_info = psutil.disk_usage(partition.mountpoint)
            storage_total += disk_info.total
            storage_used += disk_info.used
            storage_free += disk_info.free

        storage_usage = (storage_used / storage_total)*100

        storage_info = {
            'storage_total':storage_total,
            'storage_used':storage_used,
            'storage_free':storage_free,
            'storage_usage':storage_usage
        }

        return storage_info

    def get_net_info(self):
        network_info = psutil.net_io_counters()

        bytes_sent = network_info.bytes_sent
        bytes_recv = network_info.bytes_recv
        packets_sent = network_info.packets_sent
        packets_recv = network_info.packets_recv
        errin = network_info.errin
        errout = network_info.errout
        dropin = network_info.dropin
        dropout = network_info.dropout

        interfaces = psutil.net_io_counters(pernic=True)
        time.sleep(1)
        new_interfaces = psutil.net_io_counters(pernic=True)

        utilization_list = []
        for interface, stats in interfaces.items():
            max_bandwidth = psutil.net_if_stats()[interface].speed
            if max_bandwidth > 0:
                sent_bytes = new_interfaces[interface].bytes_sent - stats.bytes_sent
                recv_bytes = new_interfaces[interface].bytes_recv - stats.bytes_recv

                actual_bandwidth = (sent_bytes + recv_bytes) * 8 / (1024**2)

                utilization = (actual_bandwidth / max_bandwidth) * 100
                utilization_list.append(utilization)
        
        sys_net_usage = sum(utilization_list)/len(utilization_list)
        net_info = {
            'bytes_sent':bytes_sent,
            'bytes_recv':bytes_recv,
            'packets_sent':packets_sent,
            'packets_recv':packets_recv,
            'errin':errin,
            'errout':errout,
            'dropin':dropin,
            'dropout':dropout,
            'net_usage':sys_net_usage
        }

        return net_info
    
    def dynamic_metrics(self):
        d = {}
        d['gpu_info'] = self.get_gpu_info()
        d['cpu_info'] = self.get_cpu_info()
        d['net_info'] = self.get_net_info()
        d['mem_info'] = self.get_mem_info()
        d['storage_info'] = self.get_storage_info()

        return d
    
    def update_dynamic_indicators(self,gpu_info,cpu_info,net_info,mem_info,storage_info):
        self.gpu_info = gpu_info
        self.cpu_info = cpu_info
        self.net_info = net_info
        self.mem_info = mem_info
        self.storage_info = storage_info

        print('name:',self.name)
        print('gpu_info:',self.gpu_info)
        print('cpu_info:',self.cpu_info)
        print('net_info:',self.net_info)
        print('mem_info:',self.mem_info)
        print('storage_info:',self.storage_info)

        busyness = (0 if self.gpu_info == {} else self.gpu_info['0']['gpu_usage'])*0.3 + self.cpu_info['cpu_usage']*0.3 + self.net_info['net_usage']*0.1 + self.mem_info['mem_usage']*0.2 + self.storage_info['storage_usage']*0.1
        print('busyness:',busyness)

        dynamic_score = self.overall_score * (1-busyness/100)
        print('dynamic_score:',dynamic_score)
        
        self.busyness = busyness
        self.dynamic_score = dynamic_score
    
class ServiceCluster:
    def __init__(self, name, service_nodes):
        self.name = name
        self.service_nodes = service_nodes

        self.score = 0

    def update_score(self):
        score = sum([sn.update_score() for sn in self.service_nodes])
        self.score = score
        return score

    def get_score(self):
        return self.score
    
class Evaluator:
    def __init__(self, service_nodes, general_weights, special_weights):
        self.service_nodes = service_nodes
        self.general_weights = [0.4,0.3,0.2,0.1]
        self.special_weights = [0.3,0.5,0.1,0.1]

    def pagerank(self,B,delta=1e-6,max_iterations=100,q=1):
        M = B.shape[1]
        B_less = B / B.sum(axis=0)
        D = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                if i != j:
                    distance = np.linalg.norm(B_less[:, i] - B_less[:, j])
                    D[i, j] = distance
        U = np.where(D != 0, 1 / (1 + D), 0)
        U_row_sums = U.sum(axis=1)
        W = np.zeros_like(U)
        for i in range(M):
            if U_row_sums[i] != 0:
                W[i, :] = U[i, :] / U_row_sums[i]
            else:
                W[i, :] = 1 / M
        W = W.T
        # Rr = np.random.uniform(-1, 1, size=(M, M))
        # A = W + (q / (M * np.linalg.norm(W))) * np.dot(Rr,W)
        A = W
        X = np.ones((M, 1))
        for it in range(max_iterations):
            R = np.dot(A,X)
            d = np.linalg.norm(R - X)
            # print(d)
            if d <= delta:
                break
            X = R
        weight = R / R.sum()
        CB = np.dot(B_less,weight)
        return CB
    
    def mapping(self,rank, min_range, max_range):
        mean_value = np.mean(rank)
        res = [x - mean_value for x in np.atleast_1d(rank)]
        res = np.array(res) / (np.linalg.norm(res) or 1)
        scores = [(x+1) * (max_range - min_range)/2 for x in res]
        # scores = [(np.sin(x*np.pi/2)+1) * (max_range - min_range)/2 for x in res]
        # scores = [round(score,0) for score in scores]
        return scores

    def get_scores(self,B):
        CB = np.squeeze(self.pagerank(B))
        scores = self.mapping(CB,0,1000)
        scores = np.array([round(score,0) for score in scores])
        return scores

    def evaluate(self):
        B_c = np.vstack([sn.compute_indicators for sn in self.service_nodes])
        B_n = np.vstack([sn.communication_indicators for sn in self.service_nodes])
        B_m = np.vstack([sn.memory_indicators for sn in self.service_nodes])
        B_s = np.vstack([sn.storage_indicators for sn in self.service_nodes])
        compute_scores = self.get_scores(B_c)
        communication_scores = self.get_scores(B_n)
        memory_scores = self.get_scores(B_m)
        storage_scores = self.get_scores(B_s)
        overall_scores = compute_scores * self.general_weights[0] + communication_scores * self.general_weights[1] + memory_scores * self.general_weights[2] + storage_scores * self.general_weights[3]
        for i,sn in enumerate(self.service_nodes):
            sn.compute_score = compute_scores[i]
            sn.communication_score = communication_scores[i]
            sn.memory_score = memory_scores[i]
            sn.storage_score = storage_scores[i]
            sn.overall_score = overall_scores[i]
            
        print('service name:\t\t',[sn.name for sn in self.service_nodes])
        print('compute_scores:\t\t',compute_scores)
        print('communication_scores:\t',communication_scores)
        print('memory_scores:\t\t',memory_scores)
        print('storage_scores:\t\t',storage_scores)
        print('overall_scores:\t\t',overall_scores)
        
        return compute_scores,communication_scores,memory_scores,storage_scores,overall_scores

    def get_dynamic_scores(self, is_general):
        if is_general:
            weights = self.general_weights
        else:
            weights = self.special_weights
            
        dynamic_scores = []
        for sn in self.service_nodes:
            overall_score = sn.compute_score * weights[0] + sn.communication_score * weights[1] + sn.memory_score * weights[2] + sn.storage_score * weights[3]
            dynamic_scores.append(overall_score * (1 - sn.busyness/100))
        return dynamic_scores

class NodeContainer:
    def __init__(self,id,idt,address,sn):
        self.id = id
        self.idt = idt
        self.address = address
        self.sn = sn
        
        self.using_cpu = False
        self.using_gpu = False