# ----------------------------
# TEST OVERVIEW
# ----------------------------

    # core
        # circuit build
            # random_append(special_set)  ok
                # special_set=google_set/ibmq_set...
            # supremacy_append  ok
            # gate() | cir()  ok
            # gate() | cir([])  ok
            # gate & [] | cir  ok
            # cir.append(gate & [])  ok
            # cir.sub_circuit  ok
            # cir.to_compositegate  ok
            # cir.add_qubit  ok
            # cir.insert(gate & compositegate_index, index)  ok
            # cir.extend  ok
            # cir.inverse  ok
            # cir.gate_decomposition  ok
            # cir.set_precision  ok
            # QFT() | cir  ok
            # IQFT() | cir  ok
            # Unitary | cir  ok
            # Perm | cir  -
            # PermFx | cir  -
            # MultiControlGate | cir  -
            # cir.fast_gates  ok
            # cir.reset_qubits  ok
            # cir.find_position  -
            # cir.get_DAG_circuit  ok
            # cir.pop  ok
            # cir.adjust  -
            # cir | cir  ok
            # compositegate | cir  ok
            # compositegate(compositegate) | cir  ok
            # compositegate | compositegate  ok

        # circuit infos
            # cir.width  ok with qiskit
            # cir.size  ok with qiskit
            # cir.depth  ok with qiskit
            # cir.count_1qubit_gate  ok
            # cir.count_2qubit_gate  ok
            # cir.count_gate_by_gatetype(gate)  ok
            # cir.count_training_gate  -
            # cir.matrix  ok test with qcda
            # cir.ancilla_qubits  ok
            # cir.gates  ok
            # cir.draw  ok
            # cir.qasm  ok
            # cir.show_detail  ok

        # compositegate
            # similar to circuit

        # gate infos
            # gate.build_gate  ok
            # gate.commutative
            # gate.expand  ok
            # gate.copy  ok
            # gate.type  ok
            # gate.matrix_type  ok
            # gate.precision  ok
            # gate.qasm_name  ok
            # gate.targes + gate.controls  ok
            # gate.targ + gate.carg ok
            # gate.params  ok
            # gate.variables  ok
            # gate.is_clifford/is_diagonal....  ok

            # InstructionSet
                # InstructionSet(gate_q2, [gate_q1])  ok test with qcda

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