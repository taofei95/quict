import numpy as np

np.set_printoptions(precision=1, threshold=np.inf, suppress=True)

from QuICT.algorithm.quantum_machine_learning.encoding import *
from QuICT.simulation.state_vector import StateVectorSimulator


simulator = StateVectorSimulator(device="GPU")
print("---------------------------Test Binary Image-------------------------------")

# bin_img = np.array([[0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 1], [1, 1, 1, 1,]])
bin_img = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0,]])
frqi = FRQI(2)
# frqi_circuit_qic = frqi(bin_img, use_qic=True)
frqi_circuit_noqic = frqi(bin_img, use_qic=False)
# frqi_qic = simulator.run(frqi_circuit_qic)
frqi_noqic = simulator.run(frqi_circuit_noqic)
# print("FRQI with and without QIC should be same:")
# print(abs(np.linalg.norm(frqi_qic - frqi_noqic)) < 1e-12)

neqr = NEQR(2)
# neqr_circuit_qic = neqr(bin_img, use_qic=True)
neqr_circuit_noqic = neqr(bin_img, use_qic=False)
# neqr_qic = simulator.run(neqr_circuit_qic)
neqr_noqic = simulator.run(neqr_circuit_noqic)
# print("NEQR with and without QIC should be same:")
# print(abs(np.linalg.norm(neqr_qic - neqr_noqic)) < 1e-12)
print("NEQR and FRQI for binary image should not be same:")
print(abs(np.linalg.norm(frqi_noqic - neqr_noqic)))
print(abs(np.linalg.norm(frqi_circuit_noqic.matrix() - neqr_circuit_noqic.matrix())))

vector = np.zeros((1, 32))
vector[0, 0] = 1
print(
    abs(
        np.linalg.norm(
            (vector @ frqi_circuit_noqic.matrix())
            - (vector @ neqr_circuit_noqic.matrix())
        )
    )
)
# print(frqi_circuit_noqic.matrix() - neqr_circuit_noqic.matrix())

# print("---------------------------Test Grayscale Image-------------------------------")

# gray_img = np.array(
#     [[0, 100, 200, 255], [255, 100, 0, 100], [0, 0, 0, 255], [255, 255, 200, 0]]
# )
# frqi = FRQI(256)
# frqi_circuit_qic = frqi(gray_img, use_qic=True)
# frqi_circuit_noqic = frqi(gray_img, use_qic=False)
# frqi_qic = simulator.run(frqi_circuit_qic)
# frqi_noqic = simulator.run(frqi_circuit_noqic)
# print("FRQI with and without QIC should be same:")
# print(abs(np.linalg.norm(frqi_qic - frqi_noqic)))
# # print(frqi_qic)
# # print(frqi_noqic)

# neqr = NEQR(256)
# neqr_circuit_qic = neqr(gray_img, use_qic=True)
# neqr_circuit_noqic = neqr(gray_img, use_qic=False)
# neqr_qic = simulator.run(neqr_circuit_qic)
# neqr_noqic = simulator.run(neqr_circuit_noqic)
# print("NEQR with and without QIC should be same:")
# print(abs(np.linalg.norm(neqr_qic - neqr_noqic)))
# # print(neqr_qic)
# # print(neqr_noqic)
