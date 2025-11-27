import sys
from ns import ns
from threading import Thread,Semaphore,Lock
from queue import Queue
import pickle
import time
import uuid

ns.cppyy.cppdef("""
            Callback<void,Ptr<const Packet>, const Address&, const Address&> pythonMakeCallback(void (*f)(Ptr<const Packet>, const Address&, const Address&))
            {
                return MakeCallback(f);
            }
                
            Ptr<Packet> pythonMakePacket(std::string str)
            {
                const uint8_t* buffer = reinterpret_cast<const uint8_t*>(str.c_str());
                uint32_t size = str.length();
                Ptr<Packet> packet = Create<Packet>(buffer,size);
                return packet;
            }
                
            std::string pythonCopyFromPakcet(Ptr<const Packet> packet)
            {
                uint8_t* buffer = new uint8_t[packet->GetSize()];
                uint32_t size = packet->CopyData(buffer, packet->GetSize());
                std::string str(reinterpret_cast<char*>(buffer), size);
                return str;
            }
        """)

class WorkManager:
    def __init__(self) -> None:
        self.lock = Lock()
        self.lock_map = {}

    def add_work(self):
        work_id = uuid.uuid4()
        work_lock = Semaphore(0)
        with self.lock:
            self.lock_map[work_id] = work_lock
        return work_id,work_lock
    
    def work_done(self,work_id):
        with self.lock:
            work_lock = self.lock_map.pop(work_id)
        work_lock.release()


class Ns3Runner:
    def __init__(self) -> None:
        self.sim_run = False
        self.req_queue = Queue()
        self.ns_thread = None
        self.send_callback = self.send_packet
        self.recv_callback = self.receive_packet
        self.ip_mapping = {
            '192.168.192.158':(0,'10.1.1.1'),
            '192.168.192.175':(1,'10.1.2.1'),
            '192.168.192.225':(2,'10.1.2.2'),
            '192.168.192.225':(3,'10.1.2.3'),
            }
        
        self.work_manager = WorkManager()

    def add_request(self,src_ip,dst_ip,body):
        work_id,work_lock = self.work_manager.add_work()
        body['work_id'] = work_id
        self.req_queue.put((src_ip,dst_ip,body))
        return work_lock

    def send_packet(self):
        src_ip,dst_ip,body = self.req_queue.get()
        src_node_id,src_address = self.ip_mapping[src_ip]
        dst_node_id,dst_address = self.ip_mapping[dst_ip]
        src_node = ns.network.NodeList.GetNode(src_node_id)
        dst_node = ns.network.NodeList.GetNode(dst_node_id)

        socket = ns.network.Socket.CreateSocket(src_node, ns.network.UdpSocketFactory.GetTypeId())
        socket.Connect(ns.network.InetSocketAddress(ns.network.Ipv4Address(dst_address),9).ConvertTo())  # 设置目标节点地址
        
        d = pickle.dumps(body).decode('latin1')
        packet = ns.cppyy.gbl.pythonMakePacket(d)
        socket.Send(packet,0)  # 发送数据包
        print('sended packet size:',packet.GetSize())
        socket.Close()

        ns.core.Simulator.Schedule(ns.core.Seconds(1),ns.core.MakeEvent(self.send_callback))

    def receive_packet(self,packet,fromAddress,localAddress):
        print('received packet size:',packet.GetSize())
        buffer = ns.cppyy.gbl.pythonCopyFromPakcet(packet)
        body = pickle.loads(buffer.encode('latin1'))
        # print('sleep...')
        # time.sleep(1)
        # print('wake up')
        work_id = body['work_id']
        self.work_manager.work_done(work_id)

    def create_network_topology(self):
        ns.core.LogComponentEnable("UdpEchoClientApplication", ns.core.LOG_LEVEL_INFO)
        ns.core.LogComponentEnable("UdpEchoServerApplication", ns.core.LOG_LEVEL_INFO)
        # ns.core.LogComponentEnableAll(ns.core.LOG_LEVEL_INFO)
        # ns.core.GlobalValue.Bind("SimulatorImplementationType", ns.core.StringValue("ns3::RealtimeSimulatorImpl"))

        p2pNodes = ns.network.NodeContainer()
        p2pNodes.Create(2)

        csmaNodes = ns.network.NodeContainer()
        csmaNodes.Add(p2pNodes.Get(1))
        csmaNodes.Create(5) ##计算节点数量

        pointToPoint = ns.point_to_point.PointToPointHelper()
        pointToPoint.SetDeviceAttribute("DataRate", ns.core.StringValue("5Mbps"))
        pointToPoint.SetChannelAttribute("Delay", ns.core.StringValue("2ms"))

        p2pDevices = pointToPoint.Install(p2pNodes)

        csma = ns.csma.CsmaHelper()
        csma.SetChannelAttribute("DataRate", ns.core.StringValue("100Mbps"))
        csma.SetChannelAttribute("Delay", ns.core.TimeValue(ns.core.NanoSeconds(6560)))

        csmaDevices = csma.Install(csmaNodes)

        stack = ns.internet.InternetStackHelper()
        stack.Install(p2pNodes.Get(0))
        stack.Install(csmaNodes)

        address = ns.internet.Ipv4AddressHelper()
        address.SetBase(ns.network.Ipv4Address("10.1.1.0"), ns.network.Ipv4Mask("255.255.255.0"))
        p2pInterfaces = address.Assign(p2pDevices)

        address.SetBase(ns.network.Ipv4Address("10.1.2.0"), ns.network.Ipv4Mask("255.255.255.0"))
        csmaInterfaces = address.Assign(csmaDevices)

        echoServer = ns.applications.UdpEchoServerHelper(9)
        serverApps = echoServer.Install(csmaNodes)
        packet = ns.network.Packet() #没有这行就会报错的谜之BUG
        for i in range(serverApps.GetN()):
            app = serverApps.Get(i)
            app.TraceConnectWithoutContext("RxWithAddresses",ns.cppyy.gbl.pythonMakeCallback(self.recv_callback))

        ns.internet.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

        pointToPoint.EnablePcapAll("second")
        csma.EnablePcap("second", csmaDevices.Get(1), True)

    def simulate_network(self):
        self.sim_run = True
        print('simulation start')
        ns.core.Simulator.Run()
        ns.core.Simulator.Destroy()
        self.sim_run = False
        print('simulation finish')

    def run_simulation(self):
        self.create_network_topology()
        ns.core.Simulator.ScheduleNow(ns.core.MakeEvent(self.send_callback))
        self.simulate_network()

    def run(self):
        self.ns_thread = Thread(target=self.run_simulation)
        self.ns_thread.start()

if __name__ == '__main__':
    ns3_runner = Ns3Runner()
    ns3_runner.run()

    for i in range(2,4):
        ns3_runner.add_request('192.168.192.158',f'192.168.124.10{i}',{'data':'xxxx'})