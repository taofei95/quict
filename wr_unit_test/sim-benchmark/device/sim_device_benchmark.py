
qubit_num = [5, 6, 7, 8, 9, 10]
gate_multiply = [5, 7, 9, 11, 13, 15]
instruction_set = ["Nom", "IONQ", "IBMQ", "USTC"]

f = open("instruction_set_simulator_device_benchmark_data.txt", 'w+')
for q_num in qubit_num:
    f.write(f"qubit_number: {q_num} \n")
    for iset in instruction_set:
        f.write(f"{iset}\n")
        circuit_folder_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../new",
                iset
            )
        for gm in gate_multiply:
            f.write(f"gate size: {q_num * gm} \n")
            for i in range(1, 6):
                file_name = f"q{q_num}-g{gm * q_num}-{iset}-{i}.qasm"
                cir = OPENQASMInterface.load_file(
                    circuit_folder_path + '/' + file_name
                ).circuit
                cir_depth = cir.depth()
                cir_opt = CommutativeOptimization().execute(cir)
                gate_num = cir_opt.size()
                cir_dep = cir_opt.depth()
                f.write(f"{i}, ori circuit depth: {cir_depth}, gate size: {gate_num}, opt circuit depth: {cir_dep} \n")
