import numpy as np

from QuICT.core.utils import GateType


class GateMatrixGenerator:
    @classmethod
    def get_matrix(cls, gate, is_get_target: bool = False, controlled_by: int = 0, special_array_generator = None):
        # Step 1: Assigned array generator
        cls._array_generator = special_array_generator if special_array_generator is not None else np

        # Step 2: Get based matrix's value
        gate_type = gate.type
        gate_precision = np.complex128 if gate.precision == "double" else np.complex64
        gate_params = gate.params
        if gate_params == 0:
            based_matrix = cls.based_matrix(gate_type, gate_precision)
        else:
            based_matrix = cls.matrix_with_param(gate_type, gate_params, gate_precision)

        if is_get_target:
            return based_matrix

        # Step 3: Depending on controlled_by, generate final matrix
        assert controlled_by >= 0, "The number of controlled qubits should be positive integer."
        if controlled_by > 0:
            # TODO: expand to 111111matrix type
            pass

        return based_matrix

    def based_matrix(self, gate_type, precision):
        if gate_type in [GateType.h, GateType.ch]:
            return self._array_generator.array([
                [1 / np.sqrt(2), 1 / np.sqrt(2)],
                [1 / np.sqrt(2), -1 / np.sqrt(2)]
            ], dtype=precision)
        elif gate_type == GateType.hy:
            return self._array_generator.array([
                [1 / np.sqrt(2), -1j / np.sqrt(2)],
                [1j / np.sqrt(2), -1 / np.sqrt(2)]
            ], dtype=precision)
        elif gate_type == GateType.s:
            return self._array_generator.array([
                [1, 0],
                [0, 1j]
            ], dtype=precision)
        elif gate_type == GateType.sdg:
            return self._array_generator.array([
                [1, 0],
                [0, -1j]
            ], dtype=precision)
        elif gate_type in [GateType.x, GateType.cx, GateType.ccx]:
            return self._array_generator.array([
                [0, 1],
                [1, 0]
            ], dtype=precision)
        elif gate_type in [GateType.y, GateType.cy]:
            return self._array_generator.array([
                [0, -1j],
                [1j, 0]
            ], dtype=precision)
        elif gate_type in [GateType.z, GateType.cz, GateType.ccz]:
            return self._array_generator.array([
                [1, 0],
                [0, -1]
            ], dtype=precision)
        elif gate_type == GateType.sx:
            return self._array_generator.array([
                [0.5 + 0.5j, 0.5 - 0.5j],
                [0.5 - 0.5j, 0.5 + 0.5j]
            ], dtype=precision)
        elif gate_type == GateType.sy:
            return self._array_generator.array([
                [1 / np.sqrt(2), -1 / np.sqrt(2)],
                [1 / np.sqrt(2), 1 / np.sqrt(2)]
            ], dtype=precision)
        elif gate_type == GateType.sw:
            return self._array_generator.array([
                [1 / np.sqrt(2), -np.sqrt(1j / 2)],
                [np.sqrt(-1j / 2), 1 / np.sqrt(2)]
            ], dtype=precision)
        elif gate_type == GateType.id:
            return self._array_generator.array([
                [1, 0],
                [0, 1]
            ], dtype=precision)
        elif gate_type == GateType.t:
            return self._array_generator.array([
                [1, 0],
                [0, 1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)]
            ], dtype=precision)
        elif gate_type == GateType.tdg:
            return self._array_generator.array([
                [1, 0],
                [0, 1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)]
            ], dtype=precision)
        elif gate_type in [GateType.swap, GateType.cswap]:
            return self._array_generator.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ], dtype=precision)
        elif gate_type == GateType.iswap:
            return self._array_generator.array([
                [1, 0, 0, 0],
                [0, 0, 1j, 0],
                [0, 1j, 0, 0],
                [0, 0, 0, 1]
            ], dtype=precision)
        elif gate_type == GateType.iswapdg:
            return self._array_generator.array([
                [1, 0, 0, 0],
                [0, 0, -1j, 0],
                [0, -1j, 0, 0],
                [0, 0, 0, 1]
            ], dtype=precision)
        elif gate_type == GateType.sqiswap:
            return self._array_generator.array([
                [1, 0, 0, 0],
                [0, 0, (1 + 1j) / np.sqrt(2), 0],
                [0, (1 + 1j) / np.sqrt(2), 0, 0],
                [0, 0, 0, 1]
            ], dtype=precision)
        else:
            raise TypeError(gate_type)

    def matrix_with_param(self, gate_type, params, precision):
        if gate_type in [GateType.u1, GateType.cu1]:
            return self._array_generator.array([
                [1, 0],
                [0, np.exp(1j * params[0])]
            ], dtype=precision)

        elif gate_type == GateType.u2:
            sqrt2 = 1 / np.sqrt(2)
            return self._array_generator.array([
                [1 * sqrt2,
                 -np.exp(1j * params[1]) * sqrt2],
                [np.exp(1j * params[0]) * sqrt2,
                 np.exp(1j * (params[0] + params[1])) * sqrt2]
            ], dtype=precision)

        elif gate_type in [GateType.u3, GateType.cu3]:
            return self._array_generator.array([
                [np.cos(params[0] / 2),
                 -np.exp(1j * params[2]) * np.sin(params[0] / 2)],
                [np.exp(1j * params[1]) * np.sin(params[0] / 2),
                 np.exp(1j * (params[1] + params[2])) * np.cos(params[0] / 2)]
            ], dtype=precision)

        elif gate_type == GateType.rx:
            cos_v = self._array_generator.cos(params[0] / 2)
            sin_v = -self._array_generator.sin(params[0] / 2)
            return self._array_generator.array([
                [cos_v, 1j * sin_v],
                [1j * sin_v, cos_v]
            ], dtype=precision)

        elif gate_type == GateType.ry:
            cos_v = self._array_generator.cos(params[0] / 2)
            sin_v = self._array_generator.sin(params[0] / 2)
            return self._array_generator.array([
                [cos_v, -sin_v],
                [sin_v, cos_v]
            ], dtype=precision)

        elif gate_type in [GateType.rz, GateType.crz, GateType.ccrz]:
            return self._array_generator.array([
                [np.exp(-params[0] / 2 * 1j), 0],
                [0, np.exp(params[0] / 2 * 1j)]
            ], dtype=precision)

        elif gate_type == GateType.phase:
            return self._array_generator.array([
                [1, 0],
                [0, np.exp(params[0] * 1j)]
            ], dtype=precision)

        elif gate_type == GateType.gphase:
            return self._array_generator.array([
                [np.exp(params[0] * 1j), 0],
                [0, np.exp(params[0] * 1j)]
            ], dtype=precision)

        elif gate_type == GateType.fsim:
            costh = np.cos(params[0])
            sinth = np.sin(params[0])
            phi = params[1]
            return self._array_generator.array([
                [1, 0, 0, 0],
                [0, costh, -1j * sinth, 0],
                [0, -1j * sinth, costh, 0],
                [0, 0, 0, np.exp(-1j * phi)]
            ], dtype=precision)

        elif gate_type == GateType.rxx:
            costh = np.cos(params[0] / 2)
            sinth = np.sin(params[0] / 2)

            return np.array([
                [costh, 0, 0, -1j * sinth],
                [0, costh, -1j * sinth, 0],
                [0, -1j * sinth, costh, 0],
                [-1j * sinth, 0, 0, costh]
            ], dtype=precision)

        elif gate_type == GateType.ryy:
            costh = np.cos(params[0] / 2)
            sinth = np.sin(params[0] / 2)

            return np.array([
                [costh, 0, 0, 1j * sinth],
                [0, costh, -1j * sinth, 0],
                [0, -1j * sinth, costh, 0],
                [1j * sinth, 0, 0, costh]
            ], dtype=precision)

        elif gate_type == GateType.rzz:
            expth = np.exp(0.5j * params[0])
            sexpth = np.exp(-0.5j * params[0])

            return np.array([
                [sexpth, 0, 0, 0],
                [0, expth, 0, 0],
                [0, 0, expth, 0],
                [0, 0, 0, sexpth]
            ], dtype=precision)
        
        elif gate_type == GateType.rzx:
            costh = np.cos(params[0] / 2)
            sinth = np.sin(params[0] / 2)

            return np.array([
                [costh, -1j * sinth, 0, 0],
                [-1j * sinth, costh, 0, 0],
                [0, 0, costh, 1j * sinth],
                [0, 0, 1j * sinth, costh]
            ], dtype=precision)

        else:
            TypeError(gate_type)


class ComplexGateBuilder:
    @classmethod
    def build_gate(cls, gate_type, parg):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate()
        if gate_type in [GateType.cu3, GateType.unitary]:
            cgate = cls.build_unitary(gate.matrix)
        elif gate_type == GateType.cu1:
            cls.build_cu1(cgate)
        elif gate_type == GateType.rxx:
            cls.build_rxx(cgate, parg)
        elif gate_type == GateType.ryy:
            cls.build_ryy(cgate, parg)
        elif gate_type == GateType.rzz:
            cls.build_rzz(cgate, parg)
        elif gate_type == GateType.rzx:
            cls.build_rzx(cgate, parg)
        elif gate_type == GateType.swap:
            cls.build_swap(cgate)
        elif gate_type == GateType.ccx:
            cls.build_ccx(cgate)
        elif gate_type == GateType.ccz:
            cls.build_ccz(cgate)
        elif gate_type == GateType.ccrz:
            cls.build_ccrz(cgate, parg)
        elif gate_type == GateType.qft:
            cls.build_qft(cgate)
        elif gate_type == GateType.iqft:
            cls.build_iqft(cgate)
        elif gate_type == GateType.cswap:
            cls.build_cswap(cgate)
        else:
            return None

        return cgate

    @staticmethod
    def build_unitary(gate_matrix):
        from QuICT.qcda.synthesis import UnitaryDecomposition

        cgate, _ = UnitaryDecomposition().execute(gate_matrix)

        return cgate

    @staticmethod
    def build_qft(cgate):
        with cgate:
            for i in range(targets):
                H & i
                for j in range(i + 1, targets):
                    CU1(2 * np.pi / (1 << j - i + 1)) & [j, i]

    @staticmethod
    def build_iqft(cgate):
        with cgate:
            for i in range(targets - 1, -1, -1):
                for j in range(targets - 1, i, -1):
                    CU1(-2 * np.pi / (1 << j - i + 1)) & [j, i]
                H & i

    @staticmethod
    def build_cu1(cgate):
        with cgate:
            CRz(gate.parg) & [0, 1]
            U1(gate.parg / 2) & 0

    @staticmethod
    def build_rxx(cgate, param):
        with cgate:
            H & 0
            H & 1
            CX & [0, 1]
            Rz(param) & 1
            CX & [0, 1]
            H & 0
            H & 1

    @staticmethod
    def build_ryy(cgate, param):
        with cgate:
            Hy & 0
            Hy & 1
            CX & [0, 1]
            Rz(param) & 1
            CX & [0, 1]
            Hy & 0
            Hy & 1

    @staticmethod
    def build_rzz(cgate, param):
        with cgate:
            CX & [0, 1]
            Rz(param) & 1
            CX & [0, 1]

    @staticmethod
    def build_rzx(cgate, param):
        with cgate:
            H & 0
            CX & [0, 1]
            Rz(param) & 1
            CX & [0, 1]
            H & 0

    @staticmethod
    def build_swap(cgate):
        with cgate:
            CX & [0, 1]
            CX & [1, 0]
            CX & [0, 1]

    @staticmethod
    def build_ccx(cgate):
        with cgate:
            H & 2
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [0, 2]
            T_dagger & 2
            CX & [0, 2]
            T & 0
            T & 2
            H & 2

    @staticmethod
    def build_ccz(cgate):
        with cgate:
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [0, 2]
            T_dagger & 2
            CX & [0, 2]
            T & 0
            T & 2

    @staticmethod
    def build_ccrz(cgate, param):
        with cgate:
            CRz(param / 2) & [1, 2]
            CX & [0, 1]
            CRz(-param/ 2) & [1, 2]
            CX & [0, 1]
            CRz(param / 2) & [0, 2]

    @staticmethod
    def build_cswap(cgate):
        with cgate:
            CX & [2, 1]
            H & 2
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [0, 2]
            T_dagger & 2
            CX & [0, 2]
            T & 0
            T & 2
            H & 2
            CX & [2, 1]


class InverseGate:
    __GATE_INVERSE_MAP = {
        GateType.s: GateType.sdg,
        GateType.sdg: GateType.s,
        GateType.sx: (GateType.rx, [-np.pi / 2]),
        GateType.sy: (GateType.ry, [-np.pi / 2]),
        GateType.sw: (GateType.u2, [3 * np.pi / 4, 5 * np.pi / 4]),
        GateType.t: GateType.tdg,
        GateType.tdg: GateType.t
    }
    __INVERSE_GATE_WITH_NEGATIVE_PARAMS = [
        GateType.u1, GateType.rx, GateType.ry, GateType.phase, GateType.gphase,
        GateType.cu1, GateType.rxx, GateType.ryy, GateType.rzz, GateType.rzx,
        GateType.rz, GateType.crz, GateType.ccrz, GateType.fsim
    ]

    @classmethod
    def get_inverse_gate(cls, gate_type, params = None):
        if params is None:
            if gate_type in cls.__GATE_INVERSE_MAP.keys():
                return cls.__GATE_INVERSE_MAP[gate_type]
        else:
            inv_params = None
            if gate_type in cls.__INVERSE_GATE_WITH_NEGATIVE_PARAMS:
                inv_params = [p * -1 for p in params]
            elif gate_type == GateType.u2:
                inv_params = [np.pi - params[1], np.pi - params[0]]
            elif gate_type in [GateType.u3, GateType.cu3]:
                inv_params = [params[0], np.pi - params[2], np.pi - params[1]]

            if inv_params is not None:
                return (gate_type, inv_params)

        return None

    @staticmethod
    def inverse_unitary_gate(matrix):
        inverse_matrix = np.asmatrix(matrix).H
        return inverse_matrix

    @staticmethod
    def inverse_perm_gate(targets: int, targs: list):
        inverse_targs = [targets - 1 - t for t in targs]

        return inverse_targs
