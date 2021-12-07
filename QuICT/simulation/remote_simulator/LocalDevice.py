from DeviceLib import Device
from LocalAgent import LocalAgent


class LocalDevice(Device):
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
        result=self.__agent.Execute(JobID[1])
        #self.DeviceJobListChange(JobID[1])
        return result
    
    def ChangeAgent(self,agent):
        self.__agent = agent
        return 

    # def Execute2(self,circuit):
    #     result=self.__agent.exe(circuit)
    #     return result