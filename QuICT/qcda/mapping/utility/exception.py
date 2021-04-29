class MappingLayoutException(Exception):
    """
    
    """
    def __init__(self):
        string = str("The control and target qubit of the gate is not adjacent on the physical device")
        Exception.__init__(self, string)

class LengthException(Exception):
    """

    """
    def __init__(self):
        string = str("The length of two objects does not match ")
        Exception.__init__(self, string)