import os
import time
from QuICT.benchmark.Simulationbenchmark import Simulationbenchmark
from QuICT.benchmark.QCDAbenchmark import QCDAbenchmark
from QuICT.core.gate import *


class QuantumProblemBenchmark:
    """ A benchmarking framework for QuICT."""

    def __init__(
        self,
        run_interface=None,
        output_path: str = "."
    ):
        """
        Args:
            run_interface(optional): this is an interface that makes a series of optimizations to the original circuit
                provided and returns the optimized circuit.
                input and output for example:

                def solve_problem_interface(circuit):
                    cir_update = function(circuit)
                    return cir_update
            output_path (str, optional): The path of the Analysis of the results.
        """
        if run_interface is not None:
            self._run_interface = run_interface
        self._output_path = os.path.abspath(output_path)

    @property
    def optimizationbenchmark(self):
        return self._qcda_run("optimization")

    @property
    def mappingbenchmark(self):
        return self._qcda_run("mapping")

    @property
    def gatetransformbenchmark(self):
        return self._qcda_run("gatetransform")

    @property
    def unitarydecompositionbenchmark(self):
        return self._qcda_run("unitarydecomposition")

    @property
    def quantumstatepreparationbenchmark(self):
        return self._qcda_run("quantumstatepreparation")

    @property
    def simulationbenchmark(self):
        return self._sim_run("simulation")

    def _get_bench_data(self, bench_func) -> list:
        """Get the circuit to be benchmarked.

        Returns:
            (List[Circuit]): Return the list of output data order by output_type.
        """
        if bench_func in [
            "optimization", "mapping", "gatetransform", "unitarydecomposition", "quantumstatepreparation"
        ]:
            bench = QCDAbenchmark()
            bench_data_list = bench.get_circuits(bench_func)
        if bench_func == "simulation":
            bench = Simulationbenchmark()
            bench_data_list = bench.get_data()

        return bench_data_list

    def _qcda_run(self, bench_func):
        """Connect real-time benchmarking to the sub-physical machine to be measured.

        Returns:
            Return the analysis of QCDAbenchmarking.
        """
        data_list = self._get_bench_data(bench_func)
        data_update_list = []
        for data in data_list:
            data_update = self._run_interface(data)
            data_update_list.append(data_update)
        if bench_func in ["quantumstatepreparation"]:
            self.evaluate(bench_func=bench_func, data_update_list=data_update_list)
        else:
            self.evaluate(bench_func, data_list, data_update_list)

    def _sim_run(self, bench_func):
        """Connect real-time benchmarking to the sub-physical machine to be measured.

        Returns:
            Return the analysis of QCDAbenchmarking.
        """
        data_list = self._get_bench_data(bench_func)
        data_update_list = []
        # first simulation
        self._run_interface(data_list[0])
        for data in data_list:
            stime = time.time()
            self._run_interface(data)
            data_update_list.append(round(time.time() - stime, 4))

        self.evaluate(bench_func=bench_func, data_update_list=data_update_list)

    def evaluate(self, bench_func, data_list: list = None, data_update_list: list = None):
        ##### init framework ######
        import matplotlib.pyplot as plt
        import pandas as pd
        import prettytable as pt

        # init table
        result_file = open(self._output_path + f'/{bench_func}_benchmark_txt_show.txt', mode='w+', encoding='utf-8')
        tb = pt.PrettyTable()

        ###### line graph ######
        if bench_func == "mapping":
            result_list = []
            index = ["width", "size", "depth", "swap gate number"]
            for i in range(len(data_list)):
                cir = data_list[i]
                cir_opt = data_update_list[i]
                bench_data = [
                    cir.width() - cir_opt.width(), cir.size() - cir_opt.size(), cir.depth() - cir_opt.depth(),
                    cir.count_gate_by_gatetype("swap") - cir_opt.count_gate_by_gatetype("swap")
                ]
                result_list.append(bench_data)
                tb.field_names = index
                tb.add_row(bench_data)
        if bench_func in ["quantumstatepreparation", "unitarydecomposition"]:
            result_list = []
            index = ["size", "depth"]
            for i in range(len(data_update_list)):
                cir_opt = data_update_list[i]
                bench_data = [cir_opt.size(), cir_opt.depth()]
                result_list.append(bench_data)
                tb.field_names = index
                tb.add_row(bench_data)
        elif bench_func in ["gatetransform", "optimization"]:
            result_list = []
            index = ["size", "depth", "1-qubit gate number", "2-qubit gate number"]
            for i in range(len(data_list)):
                cir = data_list[i]
                cir_opt = data_update_list[i]
                bench_data = [
                    cir.size() - cir_opt.size(), cir.depth() - cir_opt.depth(),
                    cir.count_1qubit_gate() - cir_opt.count_1qubit_gate(),
                    cir.count_2qubit_gate() - cir_opt.count_2qubit_gate()
                ]
                result_list.append(bench_data)
                tb.field_names = index
                tb.add_row(bench_data)
        elif bench_func == "simulation":
            result_list = data_update_list
            index = ["simulation speed"]
            for i in range(len(data_update_list)):
                bench_data = [data_update_list[i]]
                tb.field_names = index
                tb.add_row(bench_data)

        df = pd.DataFrame(result_list, columns=index)
        df.plot(kind='bar', grid=True, colormap='summer_r', stacked=True)  # 堆叠图：stacked = True

        # init line graph
        plt.title(f"quantum circuit {bench_func} benchmark")
        plt.xlabel('circuit type')
        plt.ylabel('bench data')
        plt.savefig(self._output_path + f'/{bench_func}benchmark.jpg')
        plt.show()

        ###### table txt ######
        result_file.write(str(tb))
        result_file.close()
