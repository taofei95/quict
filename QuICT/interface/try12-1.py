from LocalAgent import LocalAgent 
from LocalDevice import LocalDevice 
from RemoteAgent import QiskitAgent, QuantumLeafAgent, RemoteAgent

from QuICT.core import *
from QuICT.algorithm import Amplitude
import CreateDevice
from JobLib import *

token = '69tP9UQBT661wApDMP90Ew=='
token2 ='b0d086dd5871875391ad99a1a93e20d154fe5f84acec1e76f3062c9f03847b92d7cae45307c8ec789458d13cd88e73c003b19afa67fc2bab1bf228a472c0579f'


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

job1 = SingalAgentJob(cir1)
job2 = QuantumLeafJob(cir1,1024,'cloud_baidu_sim2_water')
job3 = QiskitJob(cir1,1024,'ibmq_qasm_simulator')
print('')

print('....LocalSingleAgent....')
device1 = CreateDevice.CreateDevice(LocalDevice,LocalAgent)
jobid=device1.SendJob(job1)
result=device1.Execute(jobid)
print('Results:',result)
print('')


print('...QuantumLeafAgent...')
device2 = CreateDevice.CreateDevice(LocalDevice,QuantumLeafAgent,token)
jobid=device2.SendJob(job2)
result=device2.Execute(jobid)
print('Results:',result['counts'])
print('')

print('...QiskitAgent...')
device3 = CreateDevice.CreateDevice(LocalDevice,QiskitAgent,token2)
jobid=device3.SendJob(job3)
result=device3.Execute(jobid)
print('Results:',result.data()['counts'])
