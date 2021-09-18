from abc import ABCMeta, abstractmethod
from JobLib import *
import logging
import logging.config
logging.config.fileConfig('logging.conf')
log_ = logging.getLogger()

class Agent(object):
    
    def __init__(self):
        self.__JobID = 0
        self.__JobList = []
        self.__TotalJob = 0
        return

    @abstractmethod
    def Execute(self, JobID):
        pass

   
    def GetJob(self,job):
        return self.AgentSaveJob(job)


    
    def SendResult(self,JobID):
        return self.AgentGetResult(JobID)
        #pass

    def __AgentGetJobID(self):
        self.__TotalJob += 1
        self.__JobID = self.__TotalJob
        return self.__JobID
    
    def AgentGetTotalJob(self):
        return self.__TotalJob
    
    def AgentSaveJob(self,job):
        jobid=self.__AgentGetJobID()
        job = {'JobID':jobid,'Context':job.context,'ExecuteState':-1,'Result':0,'DeviceType':'CPU'} #jobid ; circuit; 执行状态：-1 未执行，0 正在执行，1 执行完成, 2 结果被取走; Result 最后结果
        self.__JobList.append(job)  
        return jobid

    def AgentGetJob(self,JobID):
        L= self.__JobList.__len__()
        if L>0:
            i = 0
            while i < L:
                if self.__JobList[i]['JobID']==JobID:
                    jobcontext = self.__JobList[i]['Context']
                    return jobcontext
                i += 1
            else:
                log_.error('Agent_GetJob funct: JobID: %s is worry',JobID)
        else:
            log_.error('Agent_GetJob funct: Job list is Empty')

    def AgentDrawbackJob(self,JobID):
        L= self.__JobList.__len__()
        if L>0:
            i = 0
            while i < L:
                if self.__JobList[i]['JobID']==JobID:
                    del self.__JobList[i]
                    return True
                i += 1
            else:
                log_.error('Agent_DrawbackJob: JobID: %s is worry',JobID)
        else:
            log_.error('Agent_DrawbackJob: Job list is Empty')

    def AgentGetResult(self, JobID):   #将结果传到某一内存空间，Device去读结果
        L= self.__JobList.__len__()
        if L>0:
            i = 0
            while i < L:
                if self.__JobList[i]['JobID']==JobID and self.__JobList[i]['ExecuteState']==1:
                    self.__JobList[i]['ExecuteState'] = 2
                    result = self.__JobList[i]['Result']
                    return result
                i += 1
            else:
                log_.error('Agent_GetResult: JobID: %s is worry',JobID)
        else:
            log_.error('Agent_GetResult: Job list is Empty')
 
    def AgentSaveResult(self,JobID,result):
        L= self.__JobList.__len__()
        if L>0:
            i = 0
            while i < L:
                if self.__JobList[i]['JobID']==JobID and self.__JobList[i]['ExecuteState']==0:
                    self.__JobList[i]['Result'] = result
                    self.__JobList[i]['ExecuteState'] = 1
                    return True
                i = i+1
            else:
                log_.error('Agent_SaveResult: JobID: %s is worry',JobID)
                print([i,L])
                return False
        else:
            log_.error('Agent_SaveResult: Job list is Empty')
            return False
    
    def AgentShowJobList(self):
        print(self.__JobList)
    
    def AgentChangeExeState(self,JobID,id):
        L= self.__JobList.__len__()
        if L>0:
            i = 0
            while i < L:
                if self.__JobList[i]['JobID']==JobID:
                    self.__JobList[i]['ExecuteState']=id
                    return True
                i += 1
            else:
                log_.error('Agent_ChangeExeState: JobID: %s is worry',JobID)
                return False
        else:
            log_.error('Agent_ChangeExeState: Job list is Empty')
            return False


