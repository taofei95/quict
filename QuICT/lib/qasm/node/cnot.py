# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Modification Notice: Code revised for QuICT

"""Node for an OPENQASM CNOT statement."""

from .node import Node


class Cnot(Node):
    """Node for an OPENQASM CNOT statement.

    children[0], children[1] are id nodes if CX is inside a gate body,
    otherwise they are primary nodes.
    """

    def __init__(self, children):
        """Create the cnot node."""
        super().__init__('cnot', children, None)

    def qasm(self, prec=15):
        """Return the corresponding OPENQASM string."""
        return "CX " + self.children[0].qasm(prec) + "," + \
               self.children[1].qasm(prec) + ";"
