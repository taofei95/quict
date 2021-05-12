import os

from QuICT.core.layout import *
from QuICT.qcda.mapping import Mapping
from QuICT.qcda.mapping.utility import CouplingGraph
from QuICT.tools.interface import OPENQASMInterface

if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    dir, _ = os.path.split(file_path)
    layout = Layout.load_file(f"{dir}/ibmq_casablanca.layout")
    qc = OPENQASMInterface.load_file(f"{dir}/test_example.qasm").circuit
    transformed_circuit = Mapping.run(circuit=qc, layout=layout, init_mapping_method="anneal")
    qasm = OPENQASMInterface.load_circuit(transformed_circuit)
    qasm.output_qasm(f"{dir}/output_circuit.qasm")
    print("The original circuit size is {}. After mapping, its size is {}."
          .format(qc.circuit_size(), transformed_circuit.circuit_size()))
    CouplingGraph(coupling_graph=layout).draw(file_path=f"{dir}/coupling_graph.jpg")
    qc.draw(method="matp", filename=f"{dir}/original_circuit.jpg")
    transformed_circuit.draw(method="matp",
                             filename=f"{dir}/transformed_circuit.jpg")  # Check if the number of single qubit gates and two qubit gates(except SWAP gates) remains the same
    # print([qc.circuit_count_1qubit(), transformed_circuit.circuit_count_1qubit()] )
    # print([qc.circuit_count_2qubit(), transformed_circuit.circuit_count_2qubit()] )
