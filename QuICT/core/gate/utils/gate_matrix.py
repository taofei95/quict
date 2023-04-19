import numpy as np

from .variable import Variable
from QuICT.core.utils import GateType


class GateMatrixGenerator:
    def get_matrix(
        self,
        gate,
        precision: str = "double",
        is_get_grad: bool = False,
        is_get_target: bool = False,
        special_array_generator=None,
    ):
        # Step 1: Assigned array generator
        self._array_generator = (
            special_array_generator if special_array_generator is not None else np
        )

        # Step 2: Get based matrix's value
        gate_type = gate.type
        _precision = gate.precision if precision is None else precision
        gate_precision = np.complex128 if _precision == "double" else np.complex64
        gate_params = gate.params
        if gate_params == 0:
            if is_get_grad:
                raise ValueError("Only parameter gates are supported.")
            based_matrix = self.based_matrix(gate_type, gate_precision)
        else:
            if is_get_grad:
                based_matrix = self.grad_for_param(
                    gate_type, gate.pargs, gate_precision
                )
            else:
                based_matrix = self.matrix_with_param(
                    gate_type, gate.pargs, gate_precision
                )

        if is_get_target:
            return based_matrix

        # Step 3: Depending on controlled_by, generate final matrix
        if gate.controls > 0:
            controlled_matrix = self._array_generator.identity(
                1 << (gate.controls + gate.targets), dtype=gate_precision
            )
            target_border = 1 << gate.targets
            if isinstance(based_matrix, list):
                controlled_matrix_list = [controlled_matrix] * len(based_matrix)
                for i in range(len(controlled_matrix_list)):
                    controlled_matrix_list[i][
                        -target_border:, -target_border:
                    ] = based_matrix[i]
                return controlled_matrix_list
            else:
                controlled_matrix[-target_border:, -target_border:] = based_matrix
                return controlled_matrix

        return based_matrix

    def based_matrix(self, gate_type, precision):
        if gate_type in [GateType.h, GateType.ch]:
            return self._array_generator.array(
                [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]],
                dtype=precision,
            )
        elif gate_type == GateType.hy:
            return self._array_generator.array(
                [
                    [1 / np.sqrt(2), -1j / np.sqrt(2)],
                    [1j / np.sqrt(2), -1 / np.sqrt(2)],
                ],
                dtype=precision,
            )
        elif gate_type == GateType.s:
            return self._array_generator.array([[1, 0], [0, 1j]], dtype=precision)
        elif gate_type == GateType.sdg:
            return self._array_generator.array([[1, 0], [0, -1j]], dtype=precision)
        elif gate_type in [GateType.x, GateType.cx, GateType.ccx]:
            return self._array_generator.array([[0, 1], [1, 0]], dtype=precision)
        elif gate_type in [GateType.y, GateType.cy]:
            return self._array_generator.array([[0, -1j], [1j, 0]], dtype=precision)
        elif gate_type in [GateType.z, GateType.cz, GateType.ccz]:
            return self._array_generator.array([[1, 0], [0, -1]], dtype=precision)
        elif gate_type == GateType.sx:
            return self._array_generator.array(
                [[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]], dtype=precision
            )
        elif gate_type == GateType.sy:
            return self._array_generator.array(
                [[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]],
                dtype=precision,
            )
        elif gate_type == GateType.sw:
            return self._array_generator.array(
                [
                    [1 / np.sqrt(2), -np.sqrt(1j / 2)],
                    [np.sqrt(-1j / 2), 1 / np.sqrt(2)],
                ],
                dtype=precision,
            )
        elif gate_type == GateType.id:
            return self._array_generator.array([[1, 0], [0, 1]], dtype=precision)
        elif gate_type == GateType.t:
            return self._array_generator.array(
                [[1, 0], [0, 1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)]], dtype=precision
            )
        elif gate_type == GateType.tdg:
            return self._array_generator.array(
                [[1, 0], [0, 1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)]], dtype=precision
            )
        elif gate_type in [GateType.swap, GateType.cswap]:
            return self._array_generator.array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=precision,
            )
        elif gate_type == GateType.iswap:
            return self._array_generator.array(
                [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
                dtype=precision,
            )
        elif gate_type == GateType.iswapdg:
            return self._array_generator.array(
                [[1, 0, 0, 0], [0, 0, -1j, 0], [0, -1j, 0, 0], [0, 0, 0, 1]],
                dtype=precision,
            )
        elif gate_type == GateType.sqiswap:
            return self._array_generator.array(
                [
                    [1, 0, 0, 0],
                    [0, 0, (1 + 1j) / np.sqrt(2), 0],
                    [0, (1 + 1j) / np.sqrt(2), 0, 0],
                    [0, 0, 0, 1],
                ],
                dtype=precision,
            )
        else:
            raise TypeError(gate_type)

    def matrix_with_param(self, gate_type, gate_pargs, precision):
        n_pargs = len(gate_pargs)
        pargs = [0.0] * n_pargs
        for i in range(n_pargs):
            pargs[i] = (
                gate_pargs[i].pargs
                if isinstance(gate_pargs[i], Variable)
                else gate_pargs[i]
            )

        if gate_type in [GateType.u1, GateType.cu1]:
            return self._array_generator.array(
                [[1, 0], [0, np.exp(1j * pargs[0])]], dtype=precision
            )

        elif gate_type == GateType.u2:
            sqrt2 = 1 / np.sqrt(2)
            return self._array_generator.array(
                [
                    [1 * sqrt2, -np.exp(1j * pargs[1]) * sqrt2],
                    [
                        np.exp(1j * pargs[0]) * sqrt2,
                        np.exp(1j * (pargs[0] + pargs[1])) * sqrt2,
                    ],
                ],
                dtype=precision,
            )

        elif gate_type in [GateType.u3, GateType.cu3]:
            return self._array_generator.array(
                [
                    [
                        np.cos(pargs[0] / 2),
                        -np.exp(1j * pargs[2]) * np.sin(pargs[0] / 2),
                    ],
                    [
                        np.exp(1j * pargs[1]) * np.sin(pargs[0] / 2),
                        np.exp(1j * (pargs[1] + pargs[2])) * np.cos(pargs[0] / 2),
                    ],
                ],
                dtype=precision,
            )

        elif gate_type == GateType.rx:
            cos_v = self._array_generator.cos(pargs[0] / 2)
            sin_v = -self._array_generator.sin(pargs[0] / 2)
            return self._array_generator.array(
                [[cos_v, 1j * sin_v], [1j * sin_v, cos_v]], dtype=precision
            )

        elif gate_type in [GateType.ry, GateType.cry]:
            cos_v = self._array_generator.cos(pargs[0] / 2)
            sin_v = self._array_generator.sin(pargs[0] / 2)
            return self._array_generator.array(
                [[cos_v, -sin_v], [sin_v, cos_v]], dtype=precision
            )

        elif gate_type in [GateType.rz, GateType.crz, GateType.ccrz]:
            return self._array_generator.array(
                [[np.exp(-pargs[0] / 2 * 1j), 0], [0, np.exp(pargs[0] / 2 * 1j)]],
                dtype=precision,
            )

        elif gate_type == GateType.phase:
            return self._array_generator.array(
                [[1, 0], [0, np.exp(pargs[0] * 1j)]], dtype=precision
            )

        elif gate_type == GateType.gphase:
            return self._array_generator.array(
                [[np.exp(pargs[0] * 1j), 0], [0, np.exp(pargs[0] * 1j)]],
                dtype=precision,
            )

        elif gate_type == GateType.fsim:
            costh = np.cos(pargs[0])
            sinth = np.sin(pargs[0])
            phi = pargs[1]
            return self._array_generator.array(
                [
                    [1, 0, 0, 0],
                    [0, costh, -1j * sinth, 0],
                    [0, -1j * sinth, costh, 0],
                    [0, 0, 0, np.exp(-1j * phi)],
                ],
                dtype=precision,
            )

        elif gate_type == GateType.rxx:
            costh = np.cos(pargs[0] / 2)
            sinth = np.sin(pargs[0] / 2)

            return self._array_generator.array(
                [
                    [costh, 0, 0, -1j * sinth],
                    [0, costh, -1j * sinth, 0],
                    [0, -1j * sinth, costh, 0],
                    [-1j * sinth, 0, 0, costh],
                ],
                dtype=precision,
            )

        elif gate_type == GateType.ryy:
            costh = np.cos(pargs[0] / 2)
            sinth = np.sin(pargs[0] / 2)

            return self._array_generator.array(
                [
                    [costh, 0, 0, 1j * sinth],
                    [0, costh, -1j * sinth, 0],
                    [0, -1j * sinth, costh, 0],
                    [1j * sinth, 0, 0, costh],
                ],
                dtype=precision,
            )

        elif gate_type == GateType.rzz:
            expth = np.exp(0.5j * pargs[0])
            sexpth = np.exp(-0.5j * pargs[0])

            return self._array_generator.array(
                [
                    [sexpth, 0, 0, 0],
                    [0, expth, 0, 0],
                    [0, 0, expth, 0],
                    [0, 0, 0, sexpth],
                ],
                dtype=precision,
            )

        elif gate_type == GateType.rzx:
            costh = np.cos(pargs[0] / 2)
            sinth = np.sin(pargs[0] / 2)

            return self._array_generator.array(
                [
                    [costh, -1j * sinth, 0, 0],
                    [-1j * sinth, costh, 0, 0],
                    [0, 0, costh, 1j * sinth],
                    [0, 0, 1j * sinth, costh],
                ],
                dtype=precision,
            )

        else:
            TypeError(gate_type)

    def grad_for_param(self, gate_type, gate_pargs, precision):
        n_pargs = len(gate_pargs)
        pargs = [0.0] * n_pargs
        for i in range(n_pargs):
            pargs[i] = (
                gate_pargs[i].pargs
                if isinstance(gate_pargs[i], Variable)
                else gate_pargs[i]
            )

        if gate_type in [GateType.u1, GateType.cu1]:
            return [
                self._array_generator.array(
                    [[1, 0], [0, 1j * np.exp(1j * pargs[0])]], dtype=precision
                )
            ]

        elif gate_type == GateType.u2:
            sqrt2 = 1 / np.sqrt(2)
            grad1 = self._array_generator.array(
                [
                    [1 * sqrt2, -np.exp(1j * pargs[1]) * sqrt2],
                    [
                        1j * np.exp(1j * pargs[0]) * sqrt2,
                        1j * np.exp(1j * (pargs[0] + pargs[1])) * sqrt2,
                    ],
                ],
                dtype=precision,
            )
            grad2 = self._array_generator.array(
                [
                    [1 * sqrt2, -1j * np.exp(1j * pargs[1]) * sqrt2],
                    [
                        np.exp(1j * pargs[0]) * sqrt2,
                        1j * np.exp(1j * (pargs[0] + pargs[1])) * sqrt2,
                    ],
                ],
                dtype=precision,
            )
            return [grad1, grad2]

        elif gate_type in [GateType.u3, GateType.cu3]:
            grad1 = self._array_generator.array(
                [
                    [
                        -np.sin(pargs[0] / 2) / 2,
                        -np.exp(1j * pargs[2]) * np.cos(pargs[0] / 2) / 2,
                    ],
                    [
                        np.exp(1j * pargs[1]) * np.cos(pargs[0] / 2) / 2,
                        -np.exp(1j * (pargs[1] + pargs[2])) * np.sin(pargs[0] / 2) / 2,
                    ],
                ],
                dtype=precision,
            )
            grad2 = self._array_generator.array(
                [
                    [
                        np.cos(pargs[0] / 2),
                        -np.exp(1j * pargs[2]) * np.sin(pargs[0] / 2),
                    ],
                    [
                        1j * np.exp(1j * pargs[1]) * np.sin(pargs[0] / 2),
                        1j * np.exp(1j * (pargs[1] + pargs[2])) * np.cos(pargs[0] / 2),
                    ],
                ],
                dtype=precision,
            )
            grad3 = self._array_generator.array(
                [
                    [
                        np.cos(pargs[0] / 2),
                        -1j * np.exp(1j * pargs[2]) * np.sin(pargs[0] / 2),
                    ],
                    [
                        np.exp(1j * pargs[1]) * np.sin(pargs[0] / 2),
                        1j * np.exp(1j * (pargs[1] + pargs[2])) * np.cos(pargs[0] / 2),
                    ],
                ],
                dtype=precision,
            )
            return [grad1, grad2, grad3]

        elif gate_type == GateType.rx:
            cos_v = self._array_generator.cos(pargs[0] / 2) / 2
            sin_v = -self._array_generator.sin(pargs[0] / 2) / 2
            return [
                self._array_generator.array(
                    [[-sin_v, 1j * cos_v], [1j * cos_v, -sin_v]], dtype=precision
                )
            ]

        elif gate_type in [GateType.ry, GateType.cry]:
            cos_v = self._array_generator.cos(pargs[0] / 2) / 2
            sin_v = self._array_generator.sin(pargs[0] / 2) / 2
            return [
                self._array_generator.array(
                    [[-sin_v, -cos_v], [cos_v, -sin_v]], dtype=precision
                )
            ]

        elif gate_type in [GateType.rz, GateType.crz, GateType.ccrz]:
            return [
                self._array_generator.array(
                    [
                        [-1j * np.exp(-pargs[0] / 2 * 1j) / 2, 0],
                        [0, 1j * np.exp(pargs[0] / 2 * 1j) / 2],
                    ],
                    dtype=precision,
                )
            ]

        elif gate_type == GateType.phase:
            return [
                self._array_generator.array(
                    [[1, 0], [0, 1j * np.exp(pargs[0] * 1j)]], dtype=precision
                )
            ]

        elif gate_type == GateType.gphase:
            return [
                self._array_generator.array(
                    [[1j * np.exp(pargs[0] * 1j), 0], [0, 1j * np.exp(pargs[0] * 1j)]],
                    dtype=precision,
                )
            ]

        elif gate_type == GateType.fsim:
            costh = np.cos(pargs[0])
            sinth = np.sin(pargs[0])
            phi = pargs[1]
            grad1 = self._array_generator.array(
                [
                    [1, 0, 0, 0],
                    [0, -sinth, -1j * costh, 0],
                    [0, -1j * costh, -sinth, 0],
                    [0, 0, 0, np.exp(-1j * phi)],
                ],
                dtype=precision,
            )
            grad2 = self._array_generator.array(
                [
                    [1, 0, 0, 0],
                    [0, costh, -1j * sinth, 0],
                    [0, -1j * sinth, costh, 0],
                    [0, 0, 0, -1j * np.exp(-1j * phi)],
                ],
                dtype=precision,
            )
            return [grad1, grad2]

        elif gate_type == GateType.rxx:
            costh = np.cos(pargs[0] / 2) / 2
            sinth = np.sin(pargs[0] / 2) / 2

            return [
                self._array_generator.array(
                    [
                        [-sinth, 0, 0, -1j * costh],
                        [0, -sinth, -1j * costh, 0],
                        [0, -1j * costh, -sinth, 0],
                        [-1j * costh, 0, 0, -sinth],
                    ],
                    dtype=precision,
                )
            ]

        elif gate_type == GateType.ryy:
            costh = np.cos(pargs[0] / 2) / 2
            sinth = np.sin(pargs[0] / 2) / 2

            return [
                self._array_generator.array(
                    [
                        [-sinth, 0, 0, 1j * costh],
                        [0, -sinth, -1j * costh, 0],
                        [0, -1j * costh, -sinth, 0],
                        [1j * costh, 0, 0, -sinth],
                    ],
                    dtype=precision,
                )
            ]

        elif gate_type == GateType.rzz:
            expth = 0.5j * np.exp(0.5j * pargs[0])
            sexpth = -0.5j * np.exp(-0.5j * pargs[0])

            return [
                self._array_generator.array(
                    [
                        [sexpth, 0, 0, 0],
                        [0, expth, 0, 0],
                        [0, 0, expth, 0],
                        [0, 0, 0, sexpth],
                    ],
                    dtype=precision,
                )
            ]

        elif gate_type == GateType.rzx:
            costh = np.cos(pargs[0] / 2) / 2
            sinth = np.sin(pargs[0] / 2) / 2

            return [
                self._array_generator.array(
                    [
                        [-sinth, -1j * costh, 0, 0],
                        [-1j * costh, -sinth, 0, 0],
                        [0, 0, -sinth, 1j * costh],
                        [0, 0, 1j * costh, -sinth],
                    ],
                    dtype=precision,
                )
            ]

        else:
            TypeError(gate_type)


class ComplexGateBuilder:
    @classmethod
    def build_gate(cls, gate_type, parg, gate_matrix=None):

        if gate_type == GateType.cu3:
            # TODO: currently not correct for cu3 decomposition
            # cgate = cls.build_unitary(gate_matrix)
            return None
        elif gate_type == GateType.cu1:
            cgate = cls.build_cu1(parg)
        elif gate_type == GateType.rxx:
            cgate = cls.build_rxx(parg)
        elif gate_type == GateType.ryy:
            cgate = cls.build_ryy(parg)
        elif gate_type == GateType.rzz:
            cgate = cls.build_rzz(parg)
        elif gate_type == GateType.rzx:
            cgate = cls.build_rzx(parg)
        elif gate_type == GateType.swap:
            cgate = cls.build_swap()
        elif gate_type == GateType.ccx:
            cgate = cls.build_ccx()
        elif gate_type == GateType.ccz:
            cgate = cls.build_ccz()
        elif gate_type == GateType.ccrz:
            cgate = cls.build_ccrz(parg)
        elif gate_type == GateType.cswap:
            cgate = cls.build_cswap()
        else:
            return None

        return cgate

    @staticmethod
    def build_unitary(gate_matrix):
        from QuICT.qcda.synthesis import UnitaryDecomposition

        cgate, _ = UnitaryDecomposition(include_phase_gate=True).execute(gate_matrix)

        return cgate

    @staticmethod
    def build_cu1(parg):
        return [(GateType.crz, [0, 1], [parg]), (GateType.u1, [0], [parg / 2])]

    @staticmethod
    def build_rxx(parg):
        return [
            (GateType.h, [0], None),
            (GateType.h, [1], None),
            (GateType.cx, [0, 1], None),
            (GateType.rz, [1], [parg]),
            (GateType.cx, [0, 1], None),
            (GateType.h, [0], None),
            (GateType.h, [1], None),
        ]

    @staticmethod
    def build_ryy(parg):
        return [
            (GateType.hy, [0], None),
            (GateType.hy, [1], None),
            (GateType.cx, [0, 1], None),
            (GateType.rz, [1], [parg]),
            (GateType.cx, [0, 1], None),
            (GateType.hy, [0], None),
            (GateType.hy, [1], None),
        ]

    @staticmethod
    def build_rzz(parg):
        return [
            (GateType.cx, [0, 1], None),
            (GateType.rz, [1], [parg]),
            (GateType.cx, [0, 1], None),
        ]

    @staticmethod
    def build_rzx(parg):
        return [
            (GateType.h, [0], None),
            (GateType.cx, [0, 1], None),
            (GateType.rz, [1], [parg]),
            (GateType.cx, [0, 1], None),
            (GateType.h, [0], None),
        ]

    @staticmethod
    def build_swap():
        return [
            (GateType.cx, [0, 1], None),
            (GateType.cx, [1, 0], None),
            (GateType.cx, [0, 1], None),
        ]

    @staticmethod
    def build_ccx():
        return [
            (GateType.h, [2], None),
            (GateType.cx, [2, 1], None),
            (GateType.tdg, [1], None),
            (GateType.cx, [0, 1], None),
            (GateType.t, [1], None),
            (GateType.cx, [2, 1], None),
            (GateType.tdg, [1], None),
            (GateType.cx, [0, 1], None),
            (GateType.t, [1], None),
            (GateType.cx, [0, 2], None),
            (GateType.tdg, [2], None),
            (GateType.cx, [0, 2], None),
            (GateType.t, [0], None),
            (GateType.t, [2], None),
            (GateType.h, [2], None),
        ]

    @staticmethod
    def build_ccz():
        return [
            (GateType.cx, [2, 1], None),
            (GateType.tdg, [1], None),
            (GateType.cx, [0, 1], None),
            (GateType.t, [1], None),
            (GateType.cx, [2, 1], None),
            (GateType.tdg, [1], None),
            (GateType.cx, [0, 1], None),
            (GateType.t, [1], None),
            (GateType.cx, [0, 2], None),
            (GateType.tdg, [2], None),
            (GateType.cx, [0, 2], None),
            (GateType.t, [0], None),
            (GateType.t, [2], None),
        ]

    @staticmethod
    def build_ccrz(parg):
        return [
            (GateType.crz, [1, 2], [parg / 2]),
            (GateType.cx, [0, 1], None),
            (GateType.crz, [1, 2], [-parg / 2]),
            (GateType.cx, [0, 1], None),
            (GateType.crz, [0, 2], [parg / 2]),
        ]

    @staticmethod
    def build_cswap():
        return [
            (GateType.cx, [2, 1], None),
            (GateType.h, [2], None),
            (GateType.cx, [2, 1], None),
            (GateType.tdg, [1], None),
            (GateType.cx, [0, 1], None),
            (GateType.t, [1], None),
            (GateType.cx, [2, 1], None),
            (GateType.tdg, [1], None),
            (GateType.cx, [0, 1], None),
            (GateType.t, [1], None),
            (GateType.cx, [0, 2], None),
            (GateType.tdg, [2], None),
            (GateType.cx, [0, 2], None),
            (GateType.t, [0], None),
            (GateType.t, [2], None),
            (GateType.h, [2], None),
            (GateType.cx, [2, 1], None),
        ]


class InverseGate:
    __GATE_INVERSE_MAP = {
        GateType.s: GateType.sdg,
        GateType.sdg: GateType.s,
        GateType.sx: (GateType.rx, [-np.pi / 2]),
        GateType.sy: (GateType.ry, [-np.pi / 2]),
        GateType.sw: (GateType.u2, [3 * np.pi / 4, 5 * np.pi / 4]),
        GateType.t: GateType.tdg,
        GateType.tdg: GateType.t,
    }
    __INVERSE_GATE_WITH_NEGATIVE_PARAMS = [
        GateType.u1,
        GateType.rx,
        GateType.ry,
        GateType.cry,
        GateType.phase,
        GateType.gphase,
        GateType.cu1,
        GateType.rxx,
        GateType.ryy,
        GateType.rzz,
        GateType.rzx,
        GateType.rz,
        GateType.crz,
        GateType.ccrz,
        GateType.fsim,
    ]

    @classmethod
    def get_inverse_gate(cls, gate_type: GateType, gate_pargs: list) -> tuple:
        # Maybe need gradient graph here
        if len(gate_pargs) == 0:
            if gate_type in cls.__GATE_INVERSE_MAP.keys():
                inverse_args = cls.__GATE_INVERSE_MAP[gate_type]

                return (
                    inverse_args
                    if isinstance(inverse_args, tuple)
                    else (inverse_args, gate_pargs)
                )
        else:
            inv_params = []
            if gate_type in cls.__INVERSE_GATE_WITH_NEGATIVE_PARAMS:
                for parg in gate_pargs:
                    if isinstance(parg, Variable):
                        inv_params.append(-1.0 * parg)
                    else:
                        inv_params.append(-parg)
            elif gate_type == GateType.u2:
                for parg in gate_pargs[::-1]:
                    if isinstance(parg, Variable):
                        inv_params.append(np.pi - parg)
                    else:
                        inv_params.append(np.pi - parg)
            elif gate_type in [GateType.u3, GateType.cu3]:
                for i in [0, 2, 1]:
                    if isinstance(gate_pargs[i], Variable):
                        inv_params.append(
                            -1.0 * gate_pargs[i] + np.pi
                            if i != 0
                            else gate_pargs[i] * 1.0
                        )
                    else:
                        inv_params.append(
                            np.pi - gate_pargs[i] if i != 0 else gate_pargs[i]
                        )

            if len(inv_params) > 0:
                return (gate_type, inv_params)

        return None, gate_pargs

    @staticmethod
    def inverse_perm_gate(targets: int, targs: list):
        inverse_targs = [targets - 1 - t for t in targs]

        return inverse_targs
