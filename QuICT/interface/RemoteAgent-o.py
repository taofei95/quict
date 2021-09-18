from abc import ABCMeta
from QCompute import *
from QCompute.QPlatform.QOperation import FixedGate
from qiskit import IBMQ
from qiskit import QuantumCircuit
from qiskit import transpile
from QuICT.core.gate.gate import *
from AgentLib import Agent
from JobList import JobList
from JobLib import QuantumLeafJob

class RemoteAgent(Agent, metaclass = ABCMeta):

    def __init__(self):
        Agent.__init__(self)


class QuantumLeafAgent(RemoteAgent):

    __backends = [
            'cloud_aer_at_bd',          #simulators
            'cloud_baidu_sim2_earth',
            'cloud_baidu_sim2_heaven',
            'cloud_baidu_sim2_lake',
            'cloud_baidu_sim2_thunder',
            'cloud_baidu_sim2_water',
            'cloud_baidu_sim2_wind',
            'cloud_iopcas']             #QPU

    __gates = [
            FixedGate.ID, 
            FixedGate.X, 
            FixedGate.Y, 
            FixedGate.Z, 
            FixedGate.H, 
            FixedGate.S, 
            FixedGate.SDG, 
            FixedGate.T,
            FixedGate.TDG,
            FixedGate.CX,
            FixedGate.CY,
            FixedGate.CZ,
            FixedGate.CH,
            FixedGate.SWAP,
            FixedGate.CCX,
            FixedGate.CSWAP]

    def GetBackends():
        return QuantumLeafAgent.__backends

    def GetGates():
        gates = {}
        for gate in QuantumLeafAgent.__gates:
            gates[gate.name] = gate.getMatrix()
        return gates

    def __init__(self, token):
        RemoteAgent.__init__(self)
        self.__job_list = JobList()
        self.__token = token
        Define.hubToken = self.__token
        self.__env = QEnv()
        return
    
    def _translate_input(self, circuit):
        qubits = self.__env.Q.createList(circuit.circuit_width())
        gates = circuit.gates
        for gate in gates:
            if isinstance(gate, IDGate):
                targ = gate.targ
                FixedGate.ID(qubits[targ])
            elif isinstance(gate, XGate):
                targ = gate.targ
                FixedGate.X(qubits[targ])
            elif isinstance(gate, YGate):
                targ = gate.targ
                FixedGate.Y(qubits[targ])
            elif isinstance(gate, ZGate):
                targ = gate.targ
                FixedGate.Z(qubits[targ])
            elif isinstance(gate, HGate):
                targ = gate.targ
                FixedGate.H(qubits[targ])
            elif isinstance(gate, SGate):
                targ = gate.targ
                FixedGate.S(qubits[targ])
            elif isinstance(gate, SDaggerGate):
                targ = gate.targ
                FixedGate.SDG(qubits[targ])
            elif isinstance(gate, TGate):
                targ = gate.targ
                FixedGate.T(qubits[targ])
            elif isinstance(gate, TDaggerGate):
                targ = gate.targ
                FixedGate.TDG(qubits[targ])
            elif isinstance(gate, CXGate):
                carg = gate.carg
                targ = gate.targ
                FixedGate.CX(qubits[carg], qubits[targ])
            elif isinstance(gate, CYGate):
                carg = gate.carg
                targ = gate.targ
                FixedGate.CY(qubits[carg], qubits[targ])
            elif isinstance(gate, CZGate):
                carg = gate.carg
                targ = gate.targ
                FixedGate.CZ(qubits[carg], qubits[targ])
            elif isinstance(gate, CHGate):
                carg = gate.carg
                targ = gate.targ
                FixedGate.CH(qubits[carg], qubits[targ])
            elif isinstance(gate, SwapGate):
                targs = gate.targs
                FixedGate.SWAP(qubits[targs[0]], qubits[targs[1]])
            elif isinstance(gate, CCXGate):
                cargs = gate.cargs
                targ = gate.targ
                FixedGate.CCX(qubits[cargs[0]], qubits[cargs[1]], qubits[targ])
            elif isinstance(gate, CSwapGate):
                carg = gate.carg
                targs = gate.targs
                FixedGate.CSWAP(qubits[carg], qubits[targs[0]], qubits[targs[1]])
            elif isinstance(gate, MeasureGate):
                targ = gate.targ
                MeasureZ([qubits[targ]], [targ])
            
        return self.__env
    
    def Execute(self, jobid):
        print('--> Start Execute!...')
        cb = self.AgentChangeExeState(jobid,0)
        if cb ==True:
            context= self.AgentGetJob(jobid)
            self.__env.backend(context['Backend'])
            env=self._translate_input(context['Circuit'])
            result = env.commit(context['Shots'])
            self.AgentSaveResult(jobid,result)
            print('--> Execute successfuly!')
            return result
        else:
            return print('--> Execute fault!')
        

    def SendResult(self, jobid):
        return self.AgentGetResult(jobid)
    

class QiskitAgent(RemoteAgent):
    '''
    __backends = [
            'ibmq_qasm_simulator',
            'ibmq_kolkata',
            'ibmq_mumbai',
            'ibmq_dublin',
            'ibmq_hanoi',
            'ibmq_cairo',
            'ibmq_manhattan',
            'ibmq_brooklyn',
            'ibmq_toronto',
            'ibmq_sydney',
            'ibmq_guadalupe',
            'ibmq_casablanca',
            'ibmq_lagos',
            'ibmq_nairobi',
            'ibmq_santiago',
            'ibmq_manila',
            'ibmq_bogota',
            'ibmq_jakarta',
            'ibmq_quito',
            'ibmq_belem',
            'ibmq_lima',
            'ibmq_armonk',
            ]
            '''

    def GetBackends():
        return IBMQ.providers()

    def __init__(self,token, backend_name='ibmq_qasm_simulator',hub = 'ibm-q', group = 'open', project = 'main'):
        RemoteAgent.__init__(self)
        self.__provider = IBMQ.enable_account(token)
        self.__backend = backend_name
        self.__job_list = JobList()
        print('Default Backend is ibmq_qasm_simulator !')
        return

    def _translate_input(self, circuit,backend):
        qasm_str = circuit.qasm()
        translated_circuit = QuantumCircuit.from_qasm_str(qasm_str)
        compiled_circuit = transpile(translated_circuit,backend)
        return compiled_circuit 

    def Execute(self, jobid):
        print('--> Start Execute!...')
        cb = self.AgentChangeExeState(jobid,0)
        if cb ==True:
            context= self.AgentGetJob(jobid)
            self.__backend=self.__provider.get_backend(context['Backend'])
            compiled_circuit =self._translate_input(context['Circuit'],self.__backend)
            result = self.__backend.run(compiled_circuit, shots = context['Shots'])
            
            self.AgentSaveResult(jobid,result.result())
            print('--> Execute successfuly!')
            return result.result()
        else:
            return print('--> Execute fault!')