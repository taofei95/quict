# ----------------------------
# TEST OVERVIEW
# ----------------------------

    # core
        # circuit build
            # random_append(special_set)
                # special_set=google_set/ibmq_set...
            # supremacy_append
            # gate() | cir()
            # gate() | cir([])
            # gate & [] | cir
            # cir.append(gate & [])
            # cir.sub_circuit
            # cir.to_compositegate
            # cir.add_qubit
            # cir.insert(gate & compositegate_index, index)
            # cir.extend
            # cir.inverse
            # cir.gate_decomposition
            # cir.set_precision
            # QFT() | cir
            # IQFT() | cir
            # Unitary | cir
            # Perm | cir
            # PermFx | cir
            # MultiControlGate | cir
            # cir.fast_gates
            # cir.reset_qubits
            # cir.find_position
            # cir.get_DAG_circuit
            # cir.pop
            # cir.adjust
            # cir | cir
            # compositegate | cir
            # compositegate(compositegate) | cir
            # compositegate | compositegate
            # cir | cir

        # circuit infos
            # cir.width
            # cir.size
            # cir.depth
            # cir.count_1qubit_gate
            # cir.count_2qubit_gate
            # cir.count_gate_by_gatetype(gate)
            # cir.count_training_gate
            # cir.matrix
            # cir.get_unitary_matrix
            # cir.merge_gates
            # cir.get_gates_order_by_depth
            # cir.ancilla_qubits 
            # cir.gates
            # cir.draw
            # cir.qasm

        # compositegate
            # similar to circuit

        # gate infos
            # gate.build_gate
            # gate.commutative
            # gate.expand
            # gate.copy
            # gate.permit_element
            # gate.type
            # gate.matrix_type
            # gate.precision
            # gate.qasm_name
            # gate.grad_matrix
            # gate.targes + gate.controls
            # gate.targ + gate.carg
            # gate.params
            # gate.variables
            # gate.is_clifford/is_diagonal....

            # InstructionSet
                # InstructionSet(gate_q2, [gate_q1])

        # layout
            # Layout.add_edge
            # Layout.check_edge
            # Layout.valid_circuit
            # Layout.sub_layout
            # Layout.linear_layout
            # Layout.grid_layout
            # Layout.load_file
            # Layout.to_json
            # Layout.from_json
            # Layout.write_file
            # SupremacyLayout
            # cir.random_append(layout)
            # MCTSMapping(layout)
            # SABREMapping(layout)

        # noise
            # error = BitflipError
            # error = PhaseflipError
            # error = PauliError
            # error = DampingError
            # error = ReadoutError
            # NoiseModel.add(error)
            # NoiseModel.add_noise_for_all_qubits
            # NoiseModel.add_readout_error
            # noisecircuit = NoiseModel.transpile
            # Simulator.run(NoiseModel)

        # virtual quantum machine
            # qcda.auto_compile(vqm)
            # qubit.set_fidelity/set_preparation_fidelity...
            # Simulator.run(VirtualQuantumMachine)

        # linalg
            # cpu + gpu
            # MatrixTensorI
            # MatrixPermutation
            # VectorPermutation
            # tensor
            # dot
            # multiply

    # simulation
        # circuit
            # random all gates circuit
            # algorithmic circuit from other platform
            # mirror circuit

        # StateVectorSimulator 25qubit ↓
            # device
            # precision
            # gpu_device_id
            # sync
        # StateVectorSimulator.run(circuit)
        # StateVectorSimulator.sample(circuit)

        # DensityMatrixSimulator 15qubits ↓
            # device
            # precision
            # accumulated_mode
        # DensityMatrixSimulator.run(circuit)
        # DensityMatrixSimulator.sample(circuit)

        # UnitarySimulator
            # device
            # precision
        # UnitarySimulator.run(circuit)
        # UnitarySimulator.sample(circuit)
        
        # Simulator.run(shots)
        # Simulator.run(NoiseModel)
        # Simulator.run(VirtualQuantumMachine)