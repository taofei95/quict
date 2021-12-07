import sys
import logging
import logging.config
logging.config.fileConfig('logging.conf')
log_ = logging.getLogger()


class Job(object):
    def __init__(self,circuit,supplier,shots=1024,backend='ibmq_qasm_simulator',) :
        self.context={'Circuit':circuit}
        if (supplier == 'SingleLocal'):
            self.supplier = 'SingleLocal'
        else:
            self.context['Backend'] = backend
            self.context['Shots']=shots
            if (supplier == 'Qiskit' or supplier =='qiskit'):
                self.context['Supplier'] = 'Qiskit'
            elif (supplier == 'QuantumLeaf'):
                self.context['Supplier']= 'QuantumLeaf'
            else:
                log_.error('Quantum Supplier %s is worry (Only supply "Qiskit" & "qiskit" & "QuantumLeaf" & "SingleLocal")!',supplier)
                sys.exit()
        return

# class SingalAgentJob(Job):
#     def __init__(self,circuit):
#         Job.__init__(self,circuit)
#         print('This Job is SingalAgentJob type: ',self.__dict__)
#         return

# class QuantumLeafJob(Job):
#     def __init__(self, circuit,shots,backend):
#         Job.__init__(self,circuit)
#         self.context['Shots']=shots
#         self.context['Backend'] = backend
#         print('This Job is QuantumLeafJob type: ',self.__dict__)

# class QiskitJob(Job):
#     def __init__(self, circuit,shots,backend='ibmq_qasm_simulator'):
#         Job.__init__(self,circuit)
#         self.context['Shots']=shots
#         self.context['Backend'] = backend
#         print('This Job is QiskitJob type: ',self.__dict__)
    

