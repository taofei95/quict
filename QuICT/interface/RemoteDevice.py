from DeviceLib import Device
from RemoteAgent import QuantumLeafAgent

class RemoteDevice(Device):
    def __init__(self,magent,joblist):
        Device.__init__(self,joblist)
        self.__agent = magent
    
    def SendJob(self,job):
        listid=self.DeviceJobListAdd(job)
        agentjob = self.DeviceJobListGetJob(listid)
        jobid = self.__agent.GetJob(agentjob)
        return [listid,jobid]
    
    def GetResult(self,JobID):
        result = self.__agent.SendResult(JobID[1])
        return result
    
    def Execute(self,JobID):
        env = self.__agent.env
        result=self.__agent.Execute(JobID[1])
        #self.DeviceJobListChange(JobID[1])
        return result

    def ChangeAgent(self,agent):
        self.__agent = agent
        return 