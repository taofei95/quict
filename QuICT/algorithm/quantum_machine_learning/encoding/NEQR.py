import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.exception.algorithm import *

from .FRQI import FRQI


class NEQR(FRQI):
    r"""NEQR encoding for encoding classical image data into quantum circuits.

    For more detail, please refer to:

    References:
        `NEQR: a novel enhanced quantum representation of digital images`
        <https://link.springer.com/article/10.1007/s11128-013-0567-z>

    Note:
        For a $2^n \times 2^n$ image with a gray scale of $2^q$, the number of qubits required for NEQR is
        $2n + q$ ($2n$ position qubits and $q$ color qubits).

        By default, the first $n$ qubits are the Y-axis coordinates, and the next $n$ are the X-axis coordinates.
        The final $q$ qubits are the color qubits.

    Args:
        grayscale (int, optional): The grayscale of the input images. Defaults to 2.
    """

    def __init__(self, grayscale: int = 2):
        """Initialize an NEQR instance."""
        super(NEQR, self).__init__(grayscale)
        self._n_color_qubits = int(np.log2(grayscale))
        assert 1 << self._n_color_qubits == grayscale, EncodingError(
            "The gray scale of the image should be 2^q"
        )

    def __str__(self):
        return "NEQR(n_qubits={}, color qubits={}, grayscale={})".format(
            self._n_qubits, self._n_color_qubits, self._grayscale
        )

    def __call__(self, img, use_qic=False):
        """Call NEQR for a given image.

        Args:
            img (np.ndarray): The input image.
            use_qic (bool, optional): Whether to use Quantum Image Commpression. Defaults to False.

        Returns:
            Circuit: Quantum circuit form of the image.

        Raises:
            EncodingError: An error occurred with the input image.
        """

        img = self._img_preprocess(img, flatten=True)

        # step 1: |0> -> |H>
        circuit = Circuit(self._n_qubits)
        for qid in range(self._n_pos_qubits):
            H | circuit(qid)

        # step 2: |H> -> |I>
        if use_qic:
            neqr_circuit = CompositeGate(self._n_qubits)
            groups = self._get_groups(img)
            for i in range(self._n_color_qubits):
                sub_circuit = self._construct_qic_circuit(
                    groups[i], rotate=False, gid=i
                )
                sub_circuit | neqr_circuit
        else:
            neqr_circuit = self._construct_circuit(img, rotate=False)
        neqr_circuit | circuit
        return circuit

    def _get_groups(self, img):
        img_dict = self._get_img_dict(img, bin_key=True)
        groups = np.zeros((self._n_color_qubits, self._N), dtype=np.bool_)
        for i in range(self._n_color_qubits):
            for bin_color in img_dict.keys():
                if bin_color[i] == "1":
                    for pixel in img_dict[bin_color]:
                        groups[i][pixel] = 1
        return groups
