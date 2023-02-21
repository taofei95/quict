import numpy as np

from QuICT.core.gate import MultiControlToffoli, X


def test():
    for aux_usage in ['no_aux', 'one_clean_aux', 'one_dirty_aux', 'half_dirty_aux']:
        mct = MultiControlToffoli(aux_usage)
        for control in range(9):
            mat_mct = np.eye(1 << (control + 1))
            mat_mct[(1 << (control + 1)) - 2:, (1 << (control + 1)) - 2:] = X.matrix.real
            if control <= 2:
                gates = mct(control)
                assert np.allclose(mat_mct, gates.matrix())
                continue

            if aux_usage == 'no_aux':
                gates = mct(control)
                assert np.allclose(mat_mct, gates.matrix())
            elif aux_usage in ['one_clean_aux', 'one_dirty_aux']:
                gates = mct(control)
                mat_mct = np.kron(mat_mct, np.eye(2))
                assert np.allclose(mat_mct, gates.matrix())
            else:
                if control > 5:
                    continue
                aux = control + 1
                gates = mct(control, aux)
                mat_mct = np.kron(mat_mct, np.eye(1 << aux))
                assert np.allclose(mat_mct, gates.matrix())
