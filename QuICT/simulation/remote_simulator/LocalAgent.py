from AgentLib import Agent
from QuICT.algorithm import Amplitude
from QuICT.core import *
import logging
import logging.config

logging.config.fileConfig('logging.conf')
log_ = logging.getLogger()

class LocalAgent(Agent):

    def __init__(self) :
        Agent.__init__(self)

    def Execute(self, JobID):
        log_.info('Local Start Execute...')
        cb = self.AgentChangeExeState(JobID,0)
        if cb ==True:
            context= self.AgentGetJob(JobID)
            result = Amplitude.run(context['Circuit'])
            self.AgentSaveResult(JobID,result)
        return result
    
    #def SendResult(self,JobID):
    #    return self.AgentGetResult(JobID)
    
    #def GetJob(self,job):
    #    return self.AgentSaveJob(job)
        