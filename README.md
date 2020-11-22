# QuICT

### to review the framework, you can check

#### models

​	the main part of the framework, contains:

- _qubit.py
  - implement the quantum bit and quantum register
- _circuit.py
  - implement the quantum circuit
- _gate.py
  - implement some basic quantum gate 

#### backends

- _systemcdll.py
  - use ctype link the library "quick_operator_cdll.so" which coded by C++, It is used to calculate the amplitude of the circuit

#### algorithm

​	library for quantum algorithm like shor

- _algorithm.py
  - implement the basic class Algorithm, which is the superClass of all the quantum algorithm

There is some examples which is been refactored:

- Amplitude/Amplitude.py
- SyntheticalUnitary/SyntheticalUnitary.py

#### synthesis

​	library for oracle decompose into some basic gate

- _synthesis.py
  - implement the basic class Synthesis, which is the superClass of all the synthesis algorithm

There is some examples which is been refactored:

- MCT/MCT_one_aux.py
- MCT/MCT_Linear_Simulation.py

#### optimization

​	library for oracle decompose into some basic gate

- _optimization.py
  - implement the basic class Optimization, which is the superClass of all the optimization algorithm

There is some examples which  **isn's been refactored**:

- cnot_ancillae/cnot_ancillae.py
- alter_depth_decomposition/alter_depth_decomposition.py

#### mapping

​	library for mapping

- _mapping.py
  - implement the basic class Mapping, which is the superClass of all the mapping algorithm

