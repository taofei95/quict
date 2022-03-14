# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Modification Notice: Code revised for QuICT

"""
Template 9d_7:
.. parsed-literal::
    q_0: ──■────■─────────■────■────■────■────■───────
           │    │  ┌───┐  │    │  ┌─┴─┐  │    │  ┌───┐
    q_1: ──■────┼──┤ X ├──┼────■──┤ X ├──┼────■──┤ X ├
         ┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘┌─┴─┐┌─┴─┐└─┬─┘
    q_2: ┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├┤ X ├──■──
         └───┘└───┘     └───┘└───┘     └───┘└───┘
"""

from QuICT.core import *    # pylint: disable=unused-wildcard-import
from QuICT.core.gate import *


def template_nct_9d_7():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    circuit = Circuit(3)
    CCX | circuit([0, 1, 2])
    CX | circuit([0, 2])
    CX | circuit([2, 1])
    CX | circuit([0, 2])
    CCX | circuit([0, 1, 2])
    CCX | circuit([0, 2, 1])
    CX | circuit([0, 2])
    CCX | circuit([0, 1, 2])
    CX | circuit([2, 1])
    return circuit
