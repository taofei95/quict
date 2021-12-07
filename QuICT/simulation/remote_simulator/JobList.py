
class JobList():

    def __init__(self):
        self.__jobid =0
        self.__list=[]
        return 
    
    def AddJob(self,job):
        self.__jobid += 1
        self.__list.append({'JOBID':self.__jobid,'JOB':job})
        return self.__jobid
    
    def ReduceJob(self,jobid,enforce=True):
        for i in range(len(self.__list)):
            if(self.__list[i]['JOBID']==jobid):
                if(enforce == True):
                    self.__list[i].remove()
                    return print('-> Remove success!')
                else:
                    self.__list[i].remove()
                    return print('-> Remove success!')

        return print('Error: Not Found this Job!')
    
    def GetJob(self,jobid):
        for i in range(len(self.__list)):
            if(self.__list[i]['JOBID']==jobid):
                return self.__list[i]['JOB']
        return print('Error: Not Found this Job!')
    
    def ChangeJobState(self,jobid,state):
        for i in range(len(self.__list)):
            if(self.__list[i]['JOBID']==jobid):
                self.__list[i]['JOB']['result']=state['result']
                return print('-> Update Execute result!......')
        return print('Error: Not Found this Job!')
    
    def GetTotalNum(self):
        return self.__list.__len__
    
    def ShowJobList(self):
        return print(self.__list)