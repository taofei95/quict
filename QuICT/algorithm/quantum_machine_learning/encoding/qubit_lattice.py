from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.exception.algorithm import *


class QubitLattice:
    """Qubit encoding for encoding classical image data into quantum circuits.

    After flattening the image into a vector, one qubit corresponds to one pixel.

    Note:
        Only supports binary images.

    Args:
        data_qubits (int): The number of data qubits
    """

    def __init__(self, data_qubits: int):
        """Initialize a qubit encoder instance."""
        self._data_qubits = data_qubits

    def __call__(self, img):
        """Call qubit encoding for a given image.

        Args:
            img (np.ndarray): The input image.

        Returns:
            Circuit: Quantum circuit form of the image.

        Raises:
            EncodingError: An error occurred with the input image.
        """

        img = img.flatten()
        assert img.shape[0] == self._data_qubits, EncodingError(
            "The number of pixels should be equal to the number of data qubits."
        )
        circuit = Circuit(self._data_qubits)
        for i in range(img.shape[0]):
            if img[i] == 1:
                X | circuit(i)
        return circuit
