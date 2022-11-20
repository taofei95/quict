

class QuICTBenchmark:
    def __init__(self, circuit_type: str, output_result_type: str):
        # Initial circuit library
        self._circuit_lib = CircuitLib(circuit_type)
        self._output_type = output_result_type
        pass

    def get_circuit(
        self,
        fields: List[str],
        max_width: int,
        max_size: int,
        max_depth: int
    ):
        # Get circuit from CircuitLib
        pass

    def evaluate(self, circuit_list, result_list, output_type):
        # Evaluate all circuit in circuit list group by fields
        # Step 1: Circuits group by fields
        cirs_field_mapping: Dict[str: list] = {"field": [(circuits, result)]}

        # Step 2: Score for each fields in step 1
        for cirs_field_mapping:
        self._field_score(field, List[Tuple(circuit, result)])

        # Step 3: Show Result
        self.show_result()
        pass

    def _field_score(self, field: str, circuit_result_mapping: List[Tuple]):
        # field score
        for circuit_result_mapping:
            # Step 1: score each circuit by kl, cross
            based_score = self._circuit_score()

            # Step 2: get field score from its based_score

        # Step 3: average total field score
        pass

    def _circuit_score(self, circuit, result):
        # Step 1: simulate circuit
        # Step 2: calculate kl, cross_en, l2
        # Step 3: return result
        pass

    def _kl_cal(self, p, q):
        # calculate KL
        pass

    def _cross_en_cal(self, p, q):
        # calculate cross E
        pass
    
    def _l2_cal(self, p, q):
        # calculate L2
        pass

    def show_result(self):
        # show benchmark result
        # Graph [line, radar, ...]
        # Table
        # txt file
        pass