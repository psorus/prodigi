from .belem import BElem, register
from .bint import BInt


class BFloat(BInt):
    def __init__(self, value):
        super().__init__(value)

    @classmethod
    def ident(self):
        return "FLOAT"



register(BFloat)
