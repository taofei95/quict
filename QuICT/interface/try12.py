from LocalAgent import LocalAgent 
from LocalDevice import LocalDevice 
from RemoteAgent import QiskitAgent, QuantumLeafAgent, RemoteAgent
from QuICT.core import *
from QuICT.algorithm import Amplitude
import CreateDevice
from JobLib import *

from RemoteAgent import QuantumLeafAgent

from Conf import QConf,BaiduConf


qconf = {'ApiToken':QConf.ApiToken}
token2 = qconf['ApiToken']
baiduconf = {'ApiToken':BaiduConf.ApiToken}
token = baiduconf['ApiToken']


cir1 = Circuit(2)
H | cir1(0)
H | cir1(1)
CX | cir1([0,1])
Measure | cir1([0,1])
#cir1.draw()

cir3 = Circuit(2)
H | cir3(1)
CX | cir3([1,0])
CX | cir3([0,1])
Measure | cir3([0, 1])

cir2 = Circuit(2)
X | cir2(0)
CH | cir2([0, 1])
Measure | cir2([0, 1])
#cir2.draw()

job1 = Job(cir1,'SingleLocal')
job2 = Job(cir1,'QuantumLeaf',1024,'cloud_baidu_sim2_water')
job3 = Job(cir1,'Qiskit',1024,'ibmq_qasm_simulator')




# job1 = SingalAgentJob(cir1)
# job2 = QuantumLeafJob(cir1,1024,'cloud_baidu_sim2_water')
# job3 = QiskitJob(cir1,1024,'ibmq_qasm_simulator')
print('')

print('....LocalSingleAgent....')
#device1 = CreateDevice.CreateDevice(LocalDevice,LocalAgent)
device1 = CreateDevice.CreateDevice(LocalDevice,'Local')
jobid=device1.SendJob(job1)
result=device1.Execute(jobid)
print('Results:',result)
print('')


print('...QuantumLeafAgent...')
#device2 = CreateDevice.CreateDevice(LocalDevice,QuantumLeafAgent,token)
device2 = CreateDevice.CreateDevice(LocalDevice,'QuantumLeaf',token)
jobid=device2.SendJob(job2)
result=device2.Execute(jobid)
print('Results:',result['counts'])
print('')

print('...QiskitAgent...')
#device3 = CreateDevice.CreateDevice(LocalDevice,QiskitAgent,token2)
device3 = CreateDevice.CreateDevice(LocalDevice,'Qiskit',token2)
jobid=device3.SendJob(job3)
result=device3.Execute(jobid)
print('Results:',result.data()['counts'])
