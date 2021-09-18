from abc import ABCMeta, abstractmethod
from JobList import JobList

class Device(object):

    __TotalDeviceNum = 0

    def __init__(self,list) :
        self.__TotalDeviceNum += 1
        self.joblist=list
        

    @abstractmethod
    def SendJob(self,job):
        pass

    @abstractmethod
    def GetResult(self,JobID):
        pass

    @abstractmethod
    def Execute(self,JobID):
        pass
    
    @abstractmethod
    def ChangeAgent(self,agent):
        pass

    def DeviceJobListAdd(self,job):
        return self.joblist.AddJob(job)
    
    def DeviceJobListReduce(self,jobid,enforce=True):
        self.joblist.ReduceJob(jobid,enforce)
        return
    
    def DeviceJobListGetJob(self,jobid):
        return self.joblist.GetJob(jobid)
    
    def DeviceJobListChange(self,jobid,state):
        self.joblist.ChangeState(jobid,state)
        return
    
    def DeviceJobListTotalNum(self):
        return self.joblist.GetTotalNum()
    
    def DeviceJobListShow(self):
        return self.joblist.ShowJobList()
    
    def BindAgent(self,agent):
        self._agent =agent
        return
    
    def BindJobList(self,joblist):
        self.joblist=joblist
        return
