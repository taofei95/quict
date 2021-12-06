from LocalDevice import LocalDevice
from LocalAgent import LocalAgent
from RemoteAgent import *
from JobList import JobList
import logging
import logging.config
logging.config.fileConfig('logging.conf')
log_ = logging.getLogger()

def CreateDevice(device,agent,token=None):
    if(agent == 'Local'):
        mAgent = type("AgentCreate",(LocalAgent,),dict())
        magent = mAgent()
    elif(agent == 'QuantumLeaf'):
        mAgent = type("AgentCreate",(QuantumLeafAgent,),dict())
        magent = mAgent(token)
    elif(agent == 'Qiskit'):
        mAgent = type("AgentCreate",(QiskitAgent,),dict())
        magent = mAgent(token)
    else:
        log_.error('There is no %s supplier. Only Qiskit & QuantumLeaf & Local',agent)
    # if(token==None):
    #     mAgent = type("AgentCreate",(agent,),dict())
    #     magent = mAgent()
    # else:
    #     mAgent = type("AgentCreate",(agent,),dict())
    #     magent = mAgent(token)

    mDevice = type("DeviceCreate",(device,),dict())

    joblist = JobList()
    
    mdev = mDevice(magent,joblist)
    return mdev